"""A resident, synchronous D1 API with explicit lifecycle and native batching."""

import gc
import threading
import copy
import time
from collections.abc import Mapping, Sequence
from typing import Any, Self

from .config import RuntimeConfig

from .protocol import validate_probability_rows, validate_questions, validate_response


class D1Runtime:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._model = None
        self._torch = None
        self._handles = []
        self._closed = False
        self._poisoned = False
        self._unsafe_cuda = False
        self._lock = threading.Lock()
        self._stats = {"model_loads": 0, "requests": 0, "calls": 0}

    def load(self) -> Self:
        with self._lock:
            if self._closed or self._poisoned:
                raise RuntimeError("This runtime is closed")
            if self._model is None:
                from . import backend

                try:
                    began = time.perf_counter()
                    self._model, self._torch, self._device, self._handles, receipt = (
                        backend.load(self.config)
                    )
                except BaseException:
                    self._poisoned = True
                    raise
                self._stats.update(receipt)
                self._stats["load_seconds"] = time.perf_counter() - began
        return self

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        images: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        validate_questions(questions)
        return self._invoke([(state, questions, images)], single=True)[0]

    def system_one_batch(self, requests: Sequence[tuple]) -> list[dict[str, Any]]:
        requests = list(requests)
        if not requests:
            raise ValueError("A request batch must be nonempty")
        for request in requests:
            if len(request) not in (2, 3):
                raise ValueError("Each request is (state,questions[,images])")
            validate_questions(request[1])
        return self._invoke(requests, single=False)

    def probabilities(
        self, state: Any, questions: Sequence[Any], images: Sequence[Any] | None = None
    ) -> list[list[float]]:
        named = {str(i): q for i, q in enumerate(questions)}
        validate_questions(named)
        return self._invoke(
            [(state, named, images)], single=True, raw_probabilities=True
        )[0]

    def probabilities_batch(self, requests: Sequence[tuple]) -> list[list[list[float]]]:
        if any(len(r) not in (2, 3) for r in requests):
            raise ValueError("Each request is (state,questions[,images])")
        requests = [
            (
                r[0],
                {str(i): q for i, q in enumerate(r[1])},
                r[2] if len(r) == 3 else None,
            )
            for r in requests
        ]
        if not requests:
            raise ValueError("A request batch must be nonempty")
        for request in requests:
            validate_questions(request[1])
        return self._invoke(requests, single=False, raw_probabilities=True)

    def _invoke(self, requests, *, single, raw_probabilities=False):
        self.load()
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Use one consumer per resident D1 instance")
        attempted = False
        began = time.perf_counter()
        try:
            if self._closed or self._poisoned:
                raise RuntimeError("Runtime is closed or failed")
            from .capacity import preflight
            from .guards import dispatch_guard

            preflight(
                self._model.engine,
                requests,
                self.config,
                self.config.maximum_path_tokens or self._stats["max_path_tokens"],
            )
            attempted = True

            with (
                self._torch.inference_mode(),
                dispatch_guard(self._torch, self._device),
            ):
                if raw_probabilities:
                    native = [
                        (r[0], list(r[1].values()), r[2] if len(r) == 3 else None)
                        for r in requests
                    ]
                    responses = self._model.engine.probabilities_batch(native)
                elif single:
                    state, questions, images = requests[0]
                    responses = [self._model.system_one(state, questions, images)]
                else:
                    responses = self._model.system_one_batch(requests)
            if len(responses) != len(requests):
                raise RuntimeError("Native batch cardinality changed")
            for response, request in zip(responses, requests, strict=True):
                if raw_probabilities:
                    validate_probability_rows(response, list(request[1].values()))
                else:
                    validate_response(response, request[1])
            self._stats["calls"] += 1
            self._stats["requests"] += len(requests)
            self._stats["last_request_seconds"] = time.perf_counter() - began
            return responses
        except BaseException as error:
            if attempted:
                self._poisoned = True
                from .guards import unsafe_cuda_fault

                self._unsafe_cuda = unsafe_cuda_fault(error)
            raise
        finally:
            self._lock.release()

    def stats(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self._stats, closed=self._closed))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self._model = None
            gc.collect()
            if self._torch is not None and not self._unsafe_cuda:
                with self._torch.cuda.device(self._device):
                    self._torch.cuda.empty_cache()

    def __enter__(self) -> Self:
        return self.load()

    def __exit__(self, *args):
        self.close()

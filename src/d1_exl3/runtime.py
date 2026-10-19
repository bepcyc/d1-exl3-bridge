"""A resident, synchronous D1 API with explicit lifecycle and native batching."""

import gc
import threading
import copy
import time
import traceback
from collections.abc import Mapping, Sequence
from typing import Any, Self

from .config import RuntimeConfig

from .protocol import validate_probability_rows, validate_questions, validate_response


_lifecycle_lock = threading.Lock()
_resident_runtimes = {}
_exclusive_cleanup_failed = False


def block_process():
    global _exclusive_cleanup_failed
    with _lifecycle_lock:
        _exclusive_cleanup_failed = True


def check_process():
    with _lifecycle_lock:
        if _exclusive_cleanup_failed:
            raise RuntimeError("Native execution requires a healthy worker process")


def exception_chain(error):
    """Visit causes and contexts once, including cyclic exception chains."""
    pending = [error]
    seen = set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        yield item
        pending.extend(
            link for link in (item.__cause__, item.__context__) if link is not None
        )


def clear_exception_frames(error):
    """Release completed neural frames without retaining exception objects."""
    for item in exception_chain(error):
        traceback.clear_frames(item.__traceback__)


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
        self._load_attempted = False
        self._ownership_token = object()
        self._resources = {}
        self._stats = {"model_loads": 0, "requests": 0, "calls": 0}

    def load(self) -> Self:
        with self._lock:
            check_process()
            if self._closed or self._poisoned:
                raise RuntimeError("This runtime is closed")
            if self._model is None:
                from . import backend

                with _lifecycle_lock:
                    others = list(_resident_runtimes.values())
                    if (
                        _exclusive_cleanup_failed
                        or (others and self.config.exclusive_worker_native_cleanup)
                        or any(others)
                    ):
                        raise RuntimeError(
                            "Exclusive native cleanup requires a dedicated healthy worker process"
                        )
                    _resident_runtimes[self._ownership_token] = (
                        self.config.exclusive_worker_native_cleanup
                    )
                try:
                    self._load_attempted = True
                    began = time.perf_counter()
                    self._model, self._torch, self._device, self._handles, receipt = (
                        backend.load(self.config, resources=self._resources)
                    )
                except BaseException as error:
                    self._poisoned = True
                    from .guards import unsafe_cuda_fault

                    self._unsafe_cuda = any(
                        unsafe_cuda_fault(item) for item in exception_chain(error)
                    )
                    if self._unsafe_cuda or (
                        self._resources.get("cuda_attempted")
                        and not self.config.exclusive_worker_native_cleanup
                    ):
                        block_process()
                    clear_exception_frames(error)
                    self._finish_close(load_failed=True)
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
            check_process()
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
            from .guards import unsafe_cuda_fault

            unsafe = any(unsafe_cuda_fault(item) for item in exception_chain(error))
            if attempted or unsafe:
                self._poisoned = True
                self._unsafe_cuda = unsafe
                if self._unsafe_cuda:
                    block_process()
                clear_exception_frames(error)
                responses = None
                response = None
            raise
        finally:
            self._lock.release()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            with _lifecycle_lock:
                process_unusable = _exclusive_cleanup_failed
            current = (
                self._model is not None
                and not self._closed
                and not self._unsafe_cuda
                and not process_unusable
            )
            if current:
                cuda = self._torch.cuda
                try:
                    self._stats["cuda_memory"] = {
                        "device": str(self._device),
                        "allocated_bytes": cuda.memory_allocated(self._device),
                        "reserved_bytes": cuda.memory_reserved(self._device),
                        "peak_allocated_bytes": cuda.max_memory_allocated(self._device),
                        "peak_reserved_bytes": cuda.max_memory_reserved(self._device),
                        "sampled_at_monotonic": time.monotonic(),
                    }
                except BaseException as error:
                    from .guards import unsafe_cuda_fault

                    if any(unsafe_cuda_fault(item) for item in exception_chain(error)):
                        self._unsafe_cuda = True
                        self._poisoned = True
                        block_process()
                    raise
            result = copy.deepcopy(dict(self._stats, closed=self._closed))
            result["cuda_memory_current"] = current
            result.setdefault("cuda_memory", None)
            return result

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._finish_close()

    def _finish_close(self, *, load_failed=False):
        global _exclusive_cleanup_failed
        receipt = {
            "api": "exllamav3.util.memory.free_mem",
            "status": "NOT_REQUESTED",
            "scratch_cache_cleared": False,
            "allocated_bytes_after": None,
            "reserved_bytes_after": None,
            "cleanup_errors": [],
            "load_failed": load_failed,
        }
        propagate = None
        try:
            self._closed = True
            # No GC, imports or CUDA operations for an unopened instance.
            if not self._load_attempted:
                receipt["status"] = "NOT_LOADED"
                return
            torch = self._torch or self._resources.get("torch")
            device = getattr(
                self, "_device", self._resources.get("device", self.config.device)
            )
            cuda_attempted = self._torch is not None or self._resources.get(
                "cuda_attempted", False
            )
            with _lifecycle_lock:
                process_unusable = _exclusive_cleanup_failed
            handles = self._handles or self._resources.get("handles", [])
            for handle in handles:
                try:
                    handle.remove()
                except Exception as error:
                    receipt["cleanup_errors"].append(
                        type(error).__name__ + ": " + str(error)
                    )
                    if (
                        not self.config.exclusive_worker_native_cleanup
                        and not load_failed
                    ):
                        propagate = error
            handle = None
            handles.clear()
            self._handles.clear()
            self._model = None
            self._resources.clear()
            if self._unsafe_cuda:
                receipt["status"] = "UNSAFE_CUDA_CONTEXT"
            elif process_unusable:
                receipt["status"] = "PROCESS_UNUSABLE"
            elif not cuda_attempted:
                receipt["status"] = "PRE_CUDA_LOAD_FAILED"
            else:
                gc.collect()
                if self.config.exclusive_worker_native_cleanup:
                    with torch.cuda.device(device):
                        torch.cuda.synchronize(device)
                        torch.cuda.empty_cache()
                        from exllamav3.util.tensor import g_tensor_cache
                        from exllamav3.util.memory import free_mem

                        g_tensor_cache.drop_all()
                        receipt["scratch_cache_cleared"] = True
                        free_mem()
                        torch.cuda.synchronize(device)
                        receipt["allocated_bytes_after"] = torch.cuda.memory_allocated(
                            device
                        )
                        receipt["reserved_bytes_after"] = torch.cuda.memory_reserved(
                            device
                        )
                    receipt["status"] = (
                        "RELEASED"
                        if not receipt["cleanup_errors"]
                        and receipt["allocated_bytes_after"] == 0
                        and receipt["reserved_bytes_after"] == 0
                        else "NOT_RELEASED"
                    )
                else:
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
                    if load_failed:
                        receipt["status"] = "LOAD_FAILED_NO_RELEASE_PROOF"
        except BaseException as error:
            receipt["cleanup_errors"].append(type(error).__name__ + ": " + str(error))
            receipt["status"] = "CLEANUP_FAILED"
            if not load_failed and (
                not self.config.exclusive_worker_native_cleanup
                or not isinstance(error, Exception)
            ):
                propagate = error
        finally:
            if receipt["cleanup_errors"] and receipt["status"] not in (
                "CLEANUP_FAILED",
                "UNSAFE_CUDA_CONTEXT",
            ):
                receipt["status"] = "CLEANUP_FAILED"
            self._stats["native_unload"] = receipt
            safe = receipt["status"] in (
                "RELEASED",
                "NOT_LOADED",
                "PRE_CUDA_LOAD_FAILED",
                "NOT_REQUESTED",
            )
            with _lifecycle_lock:
                if safe:
                    _resident_runtimes.pop(self._ownership_token, None)
                else:
                    _exclusive_cleanup_failed = True

        if propagate is not None:
            raise propagate

    def __enter__(self) -> Self:
        return self.load()

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            exc.add_note("Runtime cleanup also failed: " + type(cleanup_error).__name__)

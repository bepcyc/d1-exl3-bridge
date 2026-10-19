"""Native-tokenizer admission before learned execution; never truncates inputs."""

import importlib


def check_tree(
    trunk, branches, *, path_limit, packed_limit, branch_budget, max_branches
):
    if not branches or len(branches) > max_branches:
        raise ValueError("Question branches exceed the admitted capacity")
    if any(length <= 0 or length > branch_budget for length in branches):
        raise ValueError("An individual branch exceeds the native token budget")
    if trunk + max(branches) > path_limit or trunk + sum(branches) > packed_limit:
        raise ValueError("Request exceeds the admitted path/packed token capacity")


def preflight(engine, requests, config, path_limit):
    prompt = importlib.import_module(
        type(engine).__module__.rsplit(".", 1)[0] + ".prompt"
    )
    runner = importlib.import_module(type(engine).__module__)

    def count(text):
        return len(engine.tokenizer.encode(text, add_special_tokens=False))

    singles = []
    for request in requests:
        state, named = request[:2]
        images = (request[2] if len(request) == 3 else None) or ()
        questions = [prompt.as_question(q) for q in named.values()]
        if len(questions) == 1 and not images:
            row = engine.tokenizer.encode(
                engine.render(state, questions[0]), add_special_tokens=False
            )
            if not 0 < len(row) <= min(path_limit, config.token_budget):
                raise ValueError(
                    "An individual text request exceeds its path/token budget"
                )
            singles.append(row)
            continue
        prefix = prompt.prefix_text(
            engine.tokenizer,
            state,
            engine.bos,
            engine.state_style,
            engine.system,
            engine._image_markup(len(images)) if images else "",
        )
        suffixes = [
            prompt.suffix_text(engine.tokenizer, q, engine.lead, engine.option_style)
            for q in questions
        ]
        if images:
            capped = [runner.cap_pixels(image) for image in images]
            try:
                text = prefix + suffixes[0] if len(questions) == 1 else prefix
                values = engine.processor(
                    text=[text],
                    images=[capped],
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                if any(
                    getattr(getattr(v, "device", None), "type", "cpu") != "cpu"
                    for v in values.values()
                ):
                    raise RuntimeError(
                        "Preflight processor must be classical CPU preparation"
                    )
                tiles = int(values["pixel_values"].shape[0])
                if tiles > config.maximum_vision_tiles:
                    raise ValueError(
                        "Image tiles exceed the explicitly admitted capacity"
                    )
                trunk = int(values["input_ids"].shape[1])
                if len(questions) == 1:
                    if trunk > min(path_limit, config.maximum_packed_tokens):
                        raise ValueError("Image prompt exceeds admitted token capacity")
                    continue
            finally:
                for original, value in zip(images, capped, strict=True):
                    if value is not original:
                        value.close()
        else:
            trunk = count(prefix)
        branches = [count(text) for text in suffixes]
        for chunk in engine._plan(branches):
            check_tree(
                trunk,
                [branches[i] for i in chunk],
                path_limit=path_limit,
                packed_limit=config.maximum_packed_tokens,
                branch_budget=config.token_budget,
                max_branches=config.maximum_branches,
            )
    for chunk in engine._plan([len(row) for row in singles]):
        rows = [singles[i] for i in chunk]
        shared = 0
        while (
            shared < min(map(len, rows)) - 1 and len({row[shared] for row in rows}) == 1
        ):
            shared += 1
        check_tree(
            shared,
            [len(row) - shared for row in rows],
            path_limit=path_limit,
            packed_limit=config.maximum_packed_tokens,
            branch_budget=config.token_budget,
            max_branches=config.maximum_branches,
        )

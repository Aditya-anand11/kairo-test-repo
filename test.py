"""test.py

Load a previously trained model and use it to turn user input into model output.

Two model layouts are supported out of the box:

* A Hugging Face directory (must contain ``config.json``) -> text-generation pipeline.
* Anything joblib/pickle can load (e.g. a scikit-learn model, or a
  ``(vectorizer, model)`` tuple saved together) -> ``predict``.

Usage::

    python test.py                       # interactive prompt
    python test.py --prompt "hello"      # one-shot
    python test.py --model ./my_model    # explicit path
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

DEFAULT_MODEL_PATH = "model"
EXIT_WORDS = {"exit", "quit", ":q"}

Predictor = Callable[[str], str]


def load_predictor(model_path: Path) -> Predictor:
    """Load the trained model at *model_path* and return a str -> str callable."""
    if (model_path / "config.json").is_file():
        return _load_transformers(model_path)
    return _load_joblib(model_path)


def _load_transformers(model_path: Path) -> Predictor:
    try:
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "This looks like a Hugging Face model but `transformers` is not "
            "installed. Run: pip install transformers torch"
        ) from exc

    generator = pipeline(
        "text-generation",
        model=str(model_path),
        tokenizer=str(model_path),
    )

    def predict(text: str) -> str:
        result = generator(text, max_new_tokens=128, do_sample=False)
        generated = result[0]["generated_text"]
        # Strip the prompt back off when the pipeline echoes it.
        if isinstance(generated, str) and generated.startswith(text):
            generated = generated[len(text):]
        return generated.strip() or "<empty response>"

    return predict


def _load_joblib(model_path: Path) -> Predictor:
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "Loading a pickled model requires `joblib`. Run: pip install joblib"
        ) from exc

    loaded = joblib.load(model_path)

    # Common pattern: the vectorizer and the estimator were saved together.
    if isinstance(loaded, (tuple, list)) and len(loaded) == 2:
        vectorizer, estimator = loaded

        def predict_pipeline(text: str) -> str:
            features = vectorizer.transform([text])
            return str(estimator.predict(features)[0])

        return predict_pipeline

    def predict_plain(text: str) -> str:
        return str(loaded.predict([text])[0])

    return predict_plain


def run_interactive(predict: Predictor) -> int:
    print("Model loaded. Type your input, or 'exit' to quit.")
    while True:
        try:
            user_input = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        stripped = user_input.strip()
        if stripped.lower() in EXIT_WORDS:
            break
        if not stripped:
            continue

        try:
            print(f"model> {predict(user_input)}")
        except Exception as exc:  # keep the session alive on bad input
            print(f"error: {exc}", file=sys.stderr)

    print("Bye.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a trained model against user input."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"path to the trained model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="single input to feed the model; omit to enter interactive mode",
    )
    args = parser.parse_args(argv)

    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        raise SystemExit(
            f"Model not found at '{model_path}'. Train or download it first, "
            "or point to it with --model."
        )

    predict = load_predictor(model_path)

    if args.prompt is not None:
        try:
            print(predict(args.prompt))
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    return run_interactive(predict)


if __name__ == "__main__":
    raise SystemExit(main())

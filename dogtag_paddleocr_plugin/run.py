import argparse
import json
import time
from pathlib import Path
from src.paddleocr_dogtag import run_ocr


def _resolve_image(image_input) -> Path:
    if isinstance(image_input, str):
        return Path(image_input)
    if isinstance(image_input, dict):
        if local_path := image_input.get("local_path"):
            return Path(local_path)
        if (uri := image_input.get("uri", "")).startswith("file://"):
            return Path(uri.removeprefix("file://"))
    raise ValueError("inputs.image must be a string path or object with local_path/uri")


def run_plugin(payload: dict, output_dir_override: str | None) -> dict:
    started = time.perf_counter()

    params = payload.get("parameters", {})
    image_input = payload.get("inputs", {}).get("image")
    if image_input is None:
        raise ValueError("Missing required input: inputs.image")

    ocr_result = run_ocr(
        input_path=_resolve_image(image_input),
        output_dir=Path(output_dir_override or params.get("output_dir", "./outputs")),
        lang=str(params.get("lang", "en")),
        preprocess=bool(params.get("preprocess", True)),
    )

    input_metadata = payload.get("inputs", {}).get("metadata") or {}
    extracted_metadata = {
        **input_metadata,
        "num_images_processed": len(ocr_result["images"]),
        "num_text_detections": sum(len(img["detections"]) for img in ocr_result["images"]),
    }

    result = {
        "status": "success",
        "outputs": {
            "extracted_metadata": extracted_metadata,
            "ocr_text_locations": ocr_result,
        },
        "logs": "",
        "metrics": {"runtime_seconds": round(time.perf_counter() - started, 4)},
    }
    if job_id := payload.get("job_id"):
        result["job_id"] = job_id
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input.json")
    parser.add_argument("--output", default="output.json")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    output_json = Path(args.output)
    payload = {}
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        result = run_plugin(payload=payload, output_dir_override=args.output_dir)
    except Exception as exc:
        result = {
            "status": "failed",
            "outputs": {"extracted_metadata": {}, "ocr_text_locations": {}},
            "logs": str(exc),
            "metrics": {"runtime_seconds": 0.0},
        }
        if job_id := payload.get("job_id"):
            result["job_id"] = job_id

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

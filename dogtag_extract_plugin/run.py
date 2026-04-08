from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


from src.extract_dogtag import extract_dogtags


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_image_input(payload: Dict[str, Any]) -> Path:
    image_input = payload.get("inputs", {}).get("image")
    if image_input is None:
        raise ValueError("Missing required input: inputs.image")

    if isinstance(image_input, str):
        return Path(image_input)

    if isinstance(image_input, dict):
        local_path = image_input.get("local_path")
        if local_path:
            return Path(local_path)

        uri = image_input.get("uri")
        if isinstance(uri, str) and uri.startswith("file://"):
            return Path(uri.replace("file://", "", 1))
        raise ValueError("inputs.image must include local_path (or a file:// uri)")

    raise ValueError("inputs.image must be either a string path or an object")


def run_plugin(payload: Dict[str, Any], model_path_override: str | None, output_dir_override: str | None) -> Dict[str, Any]:
    started = time.perf_counter()

    job_id = payload.get("job_id")
    params = payload.get("parameters", {})
    confidence_threshold = float(params.get("confidence_threshold", 0.5))
    model_path = model_path_override or str(params.get("model_path", "./models/best.pt"))
    output_dir = Path(output_dir_override or str(params.get("output_dir", "./outputs")))

    image_path = _resolve_image_input(payload)
    saved_paths, with_dogtags, no_dogtags = extract_dogtags(
        input_path=image_path,
        output_dir=output_dir,
        model_path=model_path,
        conf_threshold=confidence_threshold,
    )

    runtime_seconds = time.perf_counter() - started
    extracted_dogtag = [{"uri": f"file://{Path(p).resolve()}"} for p in sorted(saved_paths)]
    input_metadata = payload.get("inputs", {}).get("metadata", {})
    if not isinstance(input_metadata, dict):
        input_metadata = {}

    extracted_metadata: Dict[str, Any] = dict(input_metadata)
    extracted_metadata["num_images_with_dogtags"] = len(with_dogtags)
    extracted_metadata["num_images_without_dogtags"] = len(no_dogtags)
    extracted_metadata["num_crops"] = len(saved_paths)

    result: Dict[str, Any] = {
        "status": "success",
        "outputs": {
            "extracted_metadata": extracted_metadata,
            "extracted_dogtag": extracted_dogtag,
        },
        "logs": "",
        "metrics": {
            "runtime_seconds": round(runtime_seconds, 4),
        },
    }
    if job_id is not None:
        result["job_id"] = job_id
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dogtag extraction plugin with JSON input/output.")
    parser.add_argument("--input-json", default="input.json", help="Path to plugin input JSON payload")
    parser.add_argument("--output-json", default="output.json", help="Path to write plugin output JSON payload")
    parser.add_argument("--model-path", default=None, help="Optional override for YOLO model path")
    parser.add_argument("--output-dir", default=None, help="Optional override output directory for crops")
    args = parser.parse_args()

    input_json = Path(args.input_json)
    output_json = Path(args.output_json)

    payload: Dict[str, Any] = {}
    try:
        payload = _load_json(input_json)
        result = run_plugin(
            payload=payload,
            model_path_override=args.model_path,
            output_dir_override=args.output_dir,
        )
    except Exception as exc:
        result = {
            "status": "failed",
            "outputs": {
                "extracted_metadata": {},
                "extracted_dogtag": [],
            },
            "logs": str(exc),
            "metrics": {
                "runtime_seconds": 0.0,
            },
        }
        job_id = payload.get("job_id")
        if job_id is not None:
            result["job_id"] = job_id

    _write_json(output_json, result)


if __name__ == "__main__":
    main()

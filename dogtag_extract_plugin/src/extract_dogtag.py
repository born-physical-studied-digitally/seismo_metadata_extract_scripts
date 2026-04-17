from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DOGTAG_CLASS_ID = 0


def list_images(folder: Path) -> List[Path]:
    return sorted(
        p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _clamp_box(box_xyxy: np.ndarray, width: int, height: int) -> Tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = map(int, box_xyxy.tolist())
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _sorted_dogtag_boxes(result, width: int, height: int) -> List[Tuple[int, int, int, int]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    classes = boxes.cls.detach().cpu().numpy().astype(int)
    xyxy = boxes.xyxy.detach().cpu().numpy()

    dogtag_boxes: List[Tuple[int, int, int, int]] = []
    for class_id, box in zip(classes, xyxy):
        if class_id != DOGTAG_CLASS_ID:
            continue
        clamped = _clamp_box(box, width=width, height=height)
        if clamped is not None:
            dogtag_boxes.append(clamped)

    dogtag_boxes.sort(key=lambda b: (b[1], b[0], b[3], b[2]))
    return dogtag_boxes


def extract_dogtags_from_image(
    image_path: Path,
    model: YOLO,
    output_dir: Path,
    conf_threshold: float,
) -> List[Path]:
    pil_img = Image.open(image_path).convert("RGB")
    img_for_inference = np.array(pil_img)

    results = model.predict(
        source=img_for_inference,
        save=False,
        conf=conf_threshold,
        verbose=False,
    )

    dogtag_boxes = _sorted_dogtag_boxes(results[0], width=pil_img.width, height=pil_img.height)
    saved_paths: List[Path] = []

    for i, (x1, y1, x2, y2) in enumerate(dogtag_boxes):
        crop = pil_img.crop((x1, y1, x2, y2))
        out_path = output_dir / f"{image_path.stem}_dogtag_{i}.png"
        crop.save(out_path)
        saved_paths.append(out_path)

    return saved_paths


def extract_dogtags(
    input_path: str | Path,
    output_dir: str | Path,
    model_path: str | Path = "../models/best.pt",
    conf_threshold: float = 0.5,
) -> Tuple[List[str], List[str], List[str]]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = list_images(input_path)
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    model = YOLO(str(model_path))

    all_saved: List[Path] = []
    with_dogtags: List[str] = []
    no_dogtags: List[str] = []

    for image_path in image_paths:
        saved = extract_dogtags_from_image(
            image_path=image_path,
            model=model,
            output_dir=output_dir,
            conf_threshold=conf_threshold,
        )
        all_saved.extend(saved)
        if saved:
            with_dogtags.append(image_path.name)
        else:
            no_dogtags.append(image_path.name)

    return [str(p) for p in all_saved], with_dogtags, no_dogtags

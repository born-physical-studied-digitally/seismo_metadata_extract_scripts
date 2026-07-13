from pathlib import Path
import cv2
import numpy as np
from paddleocr import PaddleOCR

IMAGE_EXTENSIONS = {".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _preprocess(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    )
    thresh = cv2.threshold(
        cv2.GaussianBlur(gray, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    h, w = img.shape[:2]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        x, y, cw, ch = cv2.boundingRect(c)
        if cw * ch > 0.25 * w * h:
            pad = 10
            return img[max(y - pad, 0):min(y + ch + pad, h), max(x - pad, 0):min(x + cw + pad, w)]

    return img


def _detect(image: np.ndarray, ocr: PaddleOCR) -> list:
    result = ocr.predict(image)
    if not result or not result[0]:
        return []

    boxes = []
    for poly, text, conf in zip(result[0]["rec_polys"], result[0]["rec_texts"], result[0]["rec_scores"]):
        pts = np.asarray(poly, dtype=np.float64)
        if pts.size == 0 or pts.ndim < 2 or pts.shape[1] < 2:
            continue
        xs, ys = pts[:, 0], pts[:, 1]
        boxes.append({
            "box": [int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))],
            "text": text,
            "confidence": float(conf),
        })
    return boxes


def _write_extracted_text(image_path: Path, detections: list, output_dir: Path) -> Path:
    output_path = output_dir / f"{image_path.stem}.txt"
    text = "\n".join(detection["text"] for detection in detections)
    if text:
        text += "\n"
    output_path.write_text(text, encoding="utf-8")
    return output_path


def run_ocr(input_path, output_dir, lang="en", preprocess=True) -> tuple[dict, list[Path]]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang=lang,
        enable_mkldnn=True,
    )

    results = []
    saved_text_paths = []
    for image_path in image_paths:
        image = _preprocess(image_path) if preprocess else cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        detections = _detect(image, ocr)
        saved_text_paths.append(_write_extracted_text(image_path, detections, output_dir))
        results.append({
            "image": str(image_path),
            "detections": detections,
        })

    return {"images": results}, saved_text_paths

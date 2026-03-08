from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image
from ultralytics import YOLO

input_dir = "./data_subset/"
output_dir = "./output/"
model_path = "./best.pt"
conf_threshold = 0.5

IMAGE_EXTENSIONS = {".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(folder: Path) -> List[Path]:
    paths = []
    for p in Path(folder).iterdir():
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    return sorted(paths)


def extract_dogtags_from_image(
    image_path: Path,
    model: YOLO,
    output_dir: Path,
    conf_threshold: float,
) -> List[str]:
    #in our model, class 0 is the dogtag. Return list of saved paths
    pil_img = Image.open(image_path)
    if pil_img.mode != "RGB":
        img_for_inference = np.array(pil_img.convert("RGB"))
    else:
        img_for_inference = np.array(pil_img)

    results = model.predict(
        source=img_for_inference,
        save=False,
        conf=conf_threshold,
    )

    saved_paths = []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return saved_paths

    dogtag_mask = boxes.cls == 0
    xyxy_dogtag = boxes.xyxy[dogtag_mask]
    if len(xyxy_dogtag) == 0:
        return saved_paths

    for i, box in enumerate(xyxy_dogtag):
        x1, y1, x2, y2 = map(int, box)
        crop = pil_img.crop((x1, y1, x2, y2))
        out_name = f"{image_path.stem}_dogtag_{i}{image_path.suffix}"
        out_path = output_dir / out_name
        crop.save(str(out_path))
        saved_paths.append(str(out_path))

    return saved_paths


def extract_dogtags(
    input_dir: str,
    output_dir: str,
    model_path: str = "best.pt",
    conf_threshold: float = 0.25,
) -> Tuple[List[str], List[str]]:

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    model = YOLO(model_path)

    image_paths = list_images(input_dir)
    if not image_paths:
        print(f"No images found in {input_dir}")
        return [], []

    all_saved = []
    with_dogtags = []
    no_dogtags = []
    for image_path in image_paths:
        paths = extract_dogtags_from_image(
            image_path, model, output_dir, conf_threshold
        )
        all_saved.extend(paths)
        if paths:
            with_dogtags.append(image_path.name)
        else:
            no_dogtags.append(image_path.name)

    if no_dogtags:
        print(f"No dogtags found in {len(no_dogtags)} image(s):")
        for name in no_dogtags:
            print(f"  {name}")

    return all_saved, with_dogtags


if __name__ == "__main__":

    """ 
    extract dogtags -> extract dogtags from imgs -> output
    """
    paths, with_dogtags = extract_dogtags(
        input_dir=input_dir,
        output_dir=output_dir,
        model_path=model_path,
        conf_threshold=conf_threshold,
    )
    print(f"Total dogtag crops saved: {len(paths)} to {output_dir}")
    print("Images with dogtags:")
    for name in with_dogtags:
        print(f"  {name}")

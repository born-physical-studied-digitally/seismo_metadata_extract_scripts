from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tensorflow import keras

input_dir = "./extracted_dogtags/"
handwriting_output_dir = "./Handwriting/"
non_handwriting_output_dir = "./Non-Handwriting/"
model_path = "./saved_model_1003.hdf5"
threshold = 0.9

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def list_images(folder: Path) -> List[Path]:
    paths = []
    for p in Path(folder).iterdir():
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    return sorted(paths)

"""
Preprocss the image to the corect size for the model.
Image : PIL Image
target_hw is a tuple of the height and width of the image.
channels is the number of channels in the image.
Returns the preprocessed image as a numpy array.
"""
def preprocess_image(pil_img: Image.Image, target_hw: Tuple[int, int], channels: int) -> np.ndarray:
    h, w = target_hw
    img = pil_img.resize((w, h), Image.Resampling.LANCZOS).convert("RGB")
    x = np.asarray(img, dtype=np.float32) / 255.0


    x = x.mean(axis=-1, keepdims=True)
    return np.expand_dims(x, axis=0)


def predict_handwriting_mask(model, pil_img: Image.Image, threshold: float) -> Tuple[np.ndarray, bool]:
    input_shape = model.input_shape
    #input_shape is a tuple of the height, width, and channels of the model.
    model_h, model_w, model_c = input_shape[1], input_shape[2], input_shape[3]
    x = preprocess_image(pil_img, (model_h, model_w), model_c)
    y = model.predict(x, verbose=0)

    #y is the score per pixel.
    y = np.squeeze(y)
    # print("yndim: ", y.ndim)
    # print("yshape: ", y.shape[-1])


    #if score is below threshold it is handwriting (1) if not it is non handwriting (0)
    handwriting_mask = (y >= threshold).astype(np.uint8)
    #And then we will turn the 0/1 mask to black white image 
    mask_resized = Image.fromarray((handwriting_mask * 255)).resize(
        pil_img.size, Image.Resampling.NEAREST
    )
    #at this point mask_resized is a black white image of the handwriting mask.
    mask_uint8 = np.asarray(mask_resized, dtype=np.uint8)
    #this boolean just tells us if any handwriting is detected.
    return mask_uint8, bool(mask_uint8.any())


def detect_handwriting_from_image(
    image_path: Path,
    model,
    handwriting_dir: Path,
    non_handwriting_dir: Path,
    threshold: float,
) -> bool:
    pil_img = Image.open(image_path).convert("RGB")
    mask, found = predict_handwriting_mask(model, pil_img, threshold)

    original = np.asarray(pil_img, dtype=np.uint8)
    #Turn it into boolean mask 
    mask_bool = mask > 0

    # First create a white image of same size as original
    handwriting_only = np.full_like(original, 255)
    #Copy only the handwriting pixel from original to the white image. 
    handwriting_only[mask_bool] = original[mask_bool]

    # White out the handwritinng part of the original image 
    handwriting_extracted = original.copy()
    handwriting_extracted[mask_bool] = 255

    only_path = handwriting_dir / f"{image_path.stem}_handwriting_only.png"
    extracted_path = non_handwriting_dir / f"{image_path.stem}_handwriting_extracted.png"
    Image.fromarray(handwriting_only).save(only_path)
    Image.fromarray(handwriting_extracted).save(extracted_path)

    return found


def detect_handwriting(
    input_dir: str,
    handwriting_output_dir: str,
    non_handwriting_output_dir: str,
    model_path: str = "unet_1024.hdf5",
    threshold: float = 0.5,
) -> Tuple[List[str], List[str]]:
    input_dir = Path(input_dir)
    handwriting_dir = Path(handwriting_output_dir)
    non_handwriting_dir = Path(non_handwriting_output_dir)

    model = keras.models.load_model(model_path, compile=False)

    image_paths = list_images(input_dir)
    if not image_paths:
        print(f"No images found in {input_dir}")
        return [], []

    with_handwriting = []
    no_handwriting = []

    for image_path in image_paths:
        found = detect_handwriting_from_image(
            image_path=image_path,
            model=model,
            handwriting_dir=handwriting_dir,
            non_handwriting_dir=non_handwriting_dir,
            threshold=threshold,
        )
        if found:
            with_handwriting.append(image_path.name)
        else:
            no_handwriting.append(image_path.name)

    if no_handwriting:
        print(f"No handwriting found in {len(no_handwriting)} image(s):")
        for i in no_handwriting:
            print(i)

    return with_handwriting, no_handwriting


if __name__ == "__main__":
    """
    detect_handwriting -> detect_handwriting_from_image -> predict_handwriting_mask
    """
    with_handwriting, no_handwriting = detect_handwriting(
        input_dir=input_dir,
        handwriting_output_dir=handwriting_output_dir,
        non_handwriting_output_dir=non_handwriting_output_dir,
        model_path=model_path,
        threshold=threshold,
    )
    print(f"Handwriting-only images: {Path(handwriting_output_dir)}")
    print(f"Non-handwriting images: {Path(non_handwriting_output_dir)}")
    print(f"Images with handwriting: {len(with_handwriting)}")
    for i in with_handwriting:
        print(i)

import faceutils as futils
import smart_path
from pathlib import Path
from PIL import Image
import fire
import numpy as np
import tqdm
from collections import defaultdict
import pickle
import cv2

from concern.image import resize_by_max


def main(
    image_dir="/data/makeup-transfer/face-style-focused/images",
    out_dir="/data/makeup-transfer/face-style-focused/landmarks"):
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    valid_paths = defaultdict(list)
    index_dir = out_dir.parent

    for image_path in tqdm.tqdm(image_dir.rglob("*")):
        if not image_path.is_file():
            continue

        sub_dir = image_path.parent.name
        file_name = image_path.name
        out_file = out_dir.joinpath(sub_dir, file_name)

        if not out_file.parent.exists():
            out_file.parent.mkdir(parents=True, exist_ok=True)

        nparr = np.fromstring(image_path.read_bytes(), np.uint8)
        try:
            np_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except:
            continue
        h, w = np_image.shape[:2]
        np_image = resize_by_max(np.asarray(np_image), 512)
        image = Image.fromarray(np_image).convert("RGB")

        with out_file.open("wb") as writer:
            face = futils.dlib.detect(image)
            if len(face) < 1:
                continue

            valid_paths[sub_dir].append(Path(sub_dir).joinpath(file_name).as_posix())
            lms = futils.dlib.landmarks(image, face[0])
            lms = lms / np_image.shape[:2] * (h, w)
            pickle.dump(lms, writer)

    for dir_name, pathlist in valid_paths.items():
        index_dir.joinpath(dir_name + ".txt").write_text("\n".join(pathlist))

if __name__ == "__main__":
    fire.Fire(main)

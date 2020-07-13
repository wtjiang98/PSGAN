import faceutils as futils
from smart_path import smart_path
from pathlib import Path
from PIL import Image
import fire
import numpy as np
import tqdm
from collections import defaultdict
import pickle
from config import config
import dlib

from multiprocessing import Pool 
from functools import partial

import os.path as osp
detector = dlib.get_frontal_face_detector()

def fast_detect(image: Image) -> 'faces':
    # rescale image since detect face on smaller image is faster
    size = 600
    width, height = image.size
    ratio = size / max(width, height)
    resize_image = image.resize((int(width * ratio), int(height * ratio)))
    resize_faces = detector(np.asarray(resize_image), 1)

    # no face detected
    if len(resize_faces) < 1:
        return []

    # rescale face box back
    left = int(resize_faces[0].left() / ratio)
    right = int(resize_faces[0].right() / ratio)
    top = int(resize_faces[0].top() / ratio)
    bottom = int(resize_faces[0].bottom() / ratio)
    face = dlib.rectangle(left, top, right, bottom)
    return [face]


def worker(image_path, out_dir):
    try:
        image = Image.open(image_path.open("rb"))

        sub_dir = image_path.parent.name
        file_name = image_path.name
        out_file = out_dir.joinpath(sub_dir, file_name)

        if not out_file.parent.exists():
            out_file.parent.mkdir(parents=True, exist_ok=True)

        with out_file.open("wb") as writer:
            face = fast_detect(image)
            if len(face) < 1:
                return

            face_on_image = face[0]
            face_image, *_ = futils.dlib.crop(image, face_on_image, config.up_ratio, config.down_ratio, config.width_ratio)
            face_image.save(writer, "PNG")

    except:
        print('Exception(Image.open) at: {}'.format(image_path.name))


def main(
    image_dir="/data/makeup-transfer/face-style/",
    out_dir="/data/datasets/beauty/crop/makeup-transfer-oss/face-style-focused",
    show=False):
    """
    dirs can also be S3 path such as s3://a/bc/
    """
    image_dir = smart_path(image_dir)
    out_dir = smart_path(out_dir)

    partial_worker = partial(worker, out_dir = out_dir)  # worker accept two parameters: image_path and out_dir

    # multiprocessing
    pool = Pool(8)
    # TODO: tqdm bar, at now, you can look at out_dir to see progress
    pool.map(partial_worker, image_dir.rglob("*"))
    pool.close()
    pool.join()


if __name__ == "__main__":
    fire.Fire(main)


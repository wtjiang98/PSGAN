from pathlib import Path

from fire import Fire
from tools.data_reader import DataReader
from neupeak.utils import webcv2
import cv2
import numpy as np


def vis_input(prefix, image, seg, lm):
    image = np.asarray(image)
    seg = (255 / seg.max() * seg).astype(np.uint8)
    lm = lm.reshape(-1, 2).round().astype(np.int32)[:, ::-1]
    for point in lm:
        image = cv2.circle(image, tuple(point), 2, (255, 0, 0))
    webcv2.imshow(prefix + "-image", image[..., ::-1])
    webcv2.imshow(prefix + "-seg", seg)
    webcv2.waitKey()


def main(data_dir="/data/makeup-transfer/MT-Dataset/"):
    data_reader = DataReader(data_dir)

    while True:
        makeup_input, non_makeup_input = data_reader.pick()
        vis_input("makeup", *makeup_input)
        vis_input("non-makeup", *non_makeup_input)


if __name__ == "__main__":
    Fire(main)

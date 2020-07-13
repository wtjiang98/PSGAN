from config import config, dataset_config
from dataloder import get_loader
from psgan.inference import Inference
from setup import setup_config, setup_argparser
import numpy as np
import neupeak.utils.webcv2 as cv2


args = setup_argparser().parse_args()
config = setup_config(args)
loader = get_loader(config)
inference = Inference(config)

for source_input, reference_input in loader:

    ret = inference.solver.test(*source_input, *reference_input)

    source = (source_input[0].squeeze(0).squeeze(
        0).numpy().transpose(1, 2, 0) + 1) / 2
    reference = (reference_input[0].squeeze(0).squeeze(
        0).numpy().transpose(1, 2, 0) + 1) / 2
    mask_s = (source_input[1][0, :, 0].numpy().transpose(
        1, 2, 0) * 255 + 0.5).astype(np.uint8)
    mask_r = (reference_input[1][0, :, 0].squeeze(
        2).numpy().transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)
    cv2.imshow("source", source[..., ::-1])
    cv2.imshow("reference", reference[..., ::-1])
    cv2.imshow("mask_s", mask_s)
    cv2.imshow("mask_r", mask_r)
    cv2.imshow("ret", np.asarray(ret)[..., ::-1])
    cv2.waitKey()

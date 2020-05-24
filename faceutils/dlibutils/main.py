#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os.path as osp

import numpy as np
from PIL import Image
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(osp.split(osp.realpath(__file__))[0] + '/lms.dat')


def detect(image: Image) -> 'faces':
    return detector(np.asarray(image), 1)


def crop(image: Image, face) -> (Image, 'face'):
    center = face.center()
    width, height = image.size
    if width > height:
        left = int(center.x - height / 2)
        right = int(center.x + height / 2)
        if left < 0:
            left, right = 0, height
        elif right > width:
            left, right = width - height, width
        image = image.crop((left, 0, right, height))
        face = dlib.rectangle(face.left() - left, face.top(), 
                              face.right() - left, face.bottom())
    elif width < height:
        top = int(center.y - width / 2)
        bottom = int(center.y + width / 2)
        if top < 0:
            top, bottom = 0, width
        elif bottom > height:
            top, bottom = height - width, height
        image = image.crop((0, top, width, bottom))
        face = dlib.rectangle(face.left(), face.top() - top, 
                              face.right(), face.bottom() - top)
    return image, face


def landmarks(image: Image, face):
    shape = predictor(np.asarray(image), face).parts()
    return np.array([[p.y, p.x] for p in shape])



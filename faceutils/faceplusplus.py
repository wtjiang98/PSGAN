#!/usr/bin/python
# -*- encoding: utf-8 -*-
from io import BytesIO
import time

import base64
import json
import requests

key = "-fd9YqPnrLnmugQGAhQoimCkQd0t8N8L"
secret = "0GLyRIHDnrjKSlDuflLPO8a6U32hyDUy"


def encode(image: 'PIL.Image') -> str:
    with BytesIO() as output_buf:
        image.save(output_buf, format='PNG')
        return base64.b64encode(output_buf.getvalue()).decode('utf-8')


def beautify(image: 'PIL.Image') -> str:
    data = {
        'api_key': key,
        'api_secret': secret,
        'image_base64': encode(image),
        }
    resp = requests.post(beautify.url, data=data)
    return resp.json()['result']


def rank(image: 'PIL.Image') -> int:
    data = {
        'api_key': key,
        'api_secret': secret,
        'image_base64': encode(image),
        'return_attributes': 'beauty',
        }
    resp = requests.post(rank.url, data=data)
    scores = resp.json()['faces'][0]['attributes']['beauty']
    return max(scores.values())


beautify.url = 'https://api-cn.faceplusplus.com/facepp/v2/beautify'
rank.url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'

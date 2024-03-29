"""
Keras RFCN
Copyright (c) 2018
Licensed under the MIT License (see LICENSE for details)
Written by parap1uie-s@github.com
"""

'''
This is a demo to Eval a RFCN model with DeepFashion Dataset
http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html
'''

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras_retinanet.models import load_model
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import cv2
import pickle

def Test(model, loadpath):
    result = open("result.csv", "w+")
    if os.path.isdir(loadpath):
        for idx, imgname in enumerate(os.listdir(loadpath)):
            if not imgname.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            print(imgname)
            image = read_image_bgr(os.path.join(loadpath, imgname))
            score = TestSinglePic(model, image, imgname)

            result.write("{},{}\n".format(imgname, score))
    result.close()

def TestSinglePic(model, image, imgname):
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    image = preprocess_image(image)
    image, scale = resize_image(image, min_side=512, max_side=512)
    boxes, scores, labels = model.predict(np.expand_dims(image, axis=0))
    return scores[0][0]

def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.

    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    ROOT_DIR = os.getcwd()
    parser = argparse.ArgumentParser()

    parser.add_argument('--loadpath', required=False,
                default="images/",
                metavar="evaluate images loadpath",
                help="evaluate images loadpath")

    args = parser.parse_args()
    model = load_model('model.h5', backbone_name='resnet50')
    Test(model, args.loadpath)
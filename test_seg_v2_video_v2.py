#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script visualize the semantic segmentation of ENet.
"""
import os
import numpy as np
from argparse import ArgumentParser
from os.path import join
import argparse
import sys
import time

print "PATH FOR IMPORTING NUMPY:"
print os.environ["PYTHONPATH"]

caffe_root = '/home/alex/repos/ENet/caffe-enet/'  # Change this to the absolute directory to ENet Caffe

import subprocess


caffe_root = '/home/alex/repos/ENet/caffe-enet/'  # Change this to the absolute directory to ENet Caffe
sys.path.insert(0, caffe_root + 'python')

os.environ["PYTHONPATH"] = "/home/alex/repos/ENet/caffe-enet/python/:/media/ubuntu/tx1_sd/repos/ENet/caffe-enet/python/caffe";
print "PATH FOR IMPORTING CAFFE:"
print os.environ["PYTHONPATH"]

import caffe
import cv2




__author__ = 'Timo Sämann'
__university__ = 'Aschaffenburg University of Applied Sciences'
__email__ = 'Timo.Saemann@gmx.de'
__data__ = '24th May, 2017'


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='.prototxt file for inference')
    parser.add_argument('--weights', type=str, required=True, help='.caffemodel file')
    parser.add_argument('--colours', type=str, required=True, help='label colours')
    parser.add_argument('--input_image', type=str, required=True, help='input image path')
    parser.add_argument('--out_dir', type=str, default=None, help='output directory in which the segmented images '
                                                                   'should be stored')
    parser.add_argument('--gpu', type=str, default='0', help='0: gpu mode active, else gpu mode inactive')

    return parser


if __name__ == '__main__':
    parser1 = make_parser()
    args = parser1.parse_args()
    if args.gpu == 0:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(args.model, args.weights, caffe.TEST)

    input_shape = net.blobs['data'].data.shape
    output_shape = net.blobs['deconv6_0_0'].data.shape

    label_colours = cv2.imread(args.colours, 1).astype(np.uint8)

    videoPath = args.input_image #"/home/alex/Downloads/20170710_153711.mp4"
    cap = cv2.VideoCapture(videoPath)

    pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
    frameskip = 30
    while True:
	
	startTime = int(round(time.time()*1000))
	flag, frame = cap.read()
	if flag:
	    # The frame is ready and already captured
	    cv2.imshow('video', frame)
	    pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
	    print str(pos_frame)+" frames"
	else:
	    # The next frame is not ready, so we try to read it again
	    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
	    print "frame is not ready"
	    # It is better to wait for a while for the next frame to be ready
	    cv2.waitKey(1000)


        if cv2.waitKey(10) == 27:
            break

        if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break

	endTime = int(round(time.time()*1000))

	print "image capture time is: " + str(endTime- startTime) + "ms"

	frameskip += 1
	if (frameskip % 10 != 0):
		continue
	
	startTime = int(round(time.time()*1000))

	input_image = cv2.resize(frame, (640,480), interpolation = cv2.INTER_CUBIC)

	input_image = cv2.resize(input_image, (input_shape[3], input_shape[2]))
	input_image = input_image.transpose((2, 0, 1))
	input_image = np.asarray([input_image])

	out = net.forward_all(**{net.inputs[0]: input_image})

	prediction = net.blobs['deconv6_0_0'].data[0].argmax(axis=0)

	prediction = np.squeeze(prediction)
	prediction = np.resize(prediction, (3, input_shape[2], input_shape[3]))
	prediction = prediction.transpose(1, 2, 0).astype(np.uint8)

	prediction_rgb = np.zeros(prediction.shape, dtype=np.uint8)
	label_colours_bgr = label_colours[..., ::-1]
	cv2.LUT(prediction, label_colours_bgr, prediction_rgb)

	endTime = int(round(time.time()*1000))
	print "processing time is: " + str(endTime- startTime) + "ms"


	startTime = int(round(time.time()*1000))
	cv2.imshow("ENet", prediction_rgb)
	endTime = int(round(time.time()*1000))

	print "display time is: " + str(endTime- startTime) + "ms"

    if args.out_dir is not None:
        input_path_ext = args.input_image.split(".")[-1]
        input_image_name = args.input_image.split("/")[-1:][0].replace('.' + input_path_ext, '')
        out_path_im = args.out_dir + input_image_name + '_enet' + '.' + input_path_ext
        out_path_gt = args.out_dir + input_image_name + '_enet_gt' + '.' + input_path_ext

        cv2.imwrite(out_path_im, prediction_rgb)
        # cv2.imwrite(out_path_gt, prediction) #  label images, where each pixel has an ID that represents the class







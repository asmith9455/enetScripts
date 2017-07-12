#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script visualize the semantic segmentation of ENet. - currently works on the TX1 at about 3 fps on the gpu for an input image of size 640x360
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

caffe_root = '/media/ubuntu/tx1_sd/repos/ENet/caffe-enet/'  # Change this to the absolute directory to ENet Caffe (TX1)
#caffe_root = '/home/alex/repos/ENet/caffe-enet/'  # Change this to the absolute directory to ENet Caffe (Desktop)

import subprocess

sys.path.insert(0, caffe_root + 'python')

os.environ["PYTHONPATH"] = caffe_root+"python/:"+caffe_root+"python/caffe/";
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

    gpuMode = False
    if args.gpu == 0 or True:
        caffe.set_mode_gpu()
	gpuMode = True
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(args.model, args.weights, caffe.TEST)

    input_shape = net.blobs['data'].data.shape
    output_shape = net.blobs['deconv6_0_0'].data.shape

    label_colours = cv2.imread(args.colours, 1).astype(np.uint8)

    videoPath = args.input_image #"/home/alex/Downloads/20170710_153711.mp4"
    cap = cv2.VideoCapture(videoPath)

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    frameskip = 100
    framectr = 0
    print "frameskip is " + str(frameskip) 
    while True:
	
	#pdb.set_trace()
	startTime = int(round(time.time()*1000))
	flag, frame = cap.read()
	framectr += 1
	if flag:
	    # The frame is ready and already captured
	    cv2.imshow('video', frame)
	    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
	    
	else:
	    # The next frame is not ready, so we try to read it again
	    cap.set(1, pos_frame-1)
	    print "frame is not ready"
	    # It is better to wait for a while for the next frame to be ready
	    cv2.waitKey(1000)


        if cv2.waitKey(10) == 27:
            break

        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break

	endTime = int(round(time.time()*1000))

	

	framectr += 1

	if (framectr % frameskip != 0):
		continue

	#pdb.set_trace()
	print str(pos_frame)+" frames"
	print "image capture time is: " + str(endTime- startTime) + "ms"
	
	startTime = int(round(time.time()*1000))
	input_image = frame
	#input_image = cv2.resize(frame, (640,480), interpolation = cv2.INTER_CUBIC)

	input_image = cv2.resize(input_image, (input_shape[3], input_shape[2]))
	input_image = input_image.transpose((2, 0, 1))
	input_image = np.asarray([input_image])

	print "prepared ip image"
	print "ip image shape is: " + str(input_image.shape)

        startTime2 = int(round(time.time()*1000))
	out = net.forward_all(**{net.inputs[0]: input_image})
	endTime2 = int(round(time.time()*1000))
	#pdb.set_trace()
	print "finished forward pass"
	prediction = net.blobs['deconv6_0_0'].data[0].argmax(axis=0)
	endTime3 = int(round(time.time()*1000))
       
	print 'finished prediction extraction'
	#pdb.set_trace()

	prediction = np.squeeze(prediction)
	prediction = np.resize(prediction, (3, input_shape[2], input_shape[3]))
	prediction = prediction.transpose(1, 2, 0).astype(np.uint8)

	prediction_rgb = np.zeros(prediction.shape, dtype=np.uint8)
	label_colours_bgr = label_colours[..., ::-1]
	cv2.LUT(prediction, label_colours_bgr, prediction_rgb)

	endTime = int(round(time.time()*1000))
	if (gpuMode):
		print "processing time is: " + str(endTime- startTime) + "ms (gpu mode)"
	else:
		print "processing time is: " + str(endTime- startTime) + "ms (cpu mode)"
	print "of this, nn fwd propagation time is: " + str(endTime2- startTime2) + "ms"
	print "of this, nn data rearrangement is: " + str(endTime3- endTime2) + "ms"


	startTime = int(round(time.time()*1000))
	cv2.imshow("ENet", prediction_rgb)

	print "prediction image shape is: " + str(prediction_rgb.shape)
	
	endTime = int(round(time.time()*1000))

	print "display time is: " + str(endTime- startTime) + "ms"

	print "not waiting for keypress"
	#cv2.waitKey(0)
	print "didn't get keypress"

    if (args.out_dir is not None) and False:
        input_path_ext = args.input_image.split(".")[-1]
        input_image_name = args.input_image.split("/")[-1:][0].replace('.' + input_path_ext, '')
        out_path_im = args.out_dir + input_image_name + '_enet' + '.' + input_path_ext
        out_path_gt = args.out_dir + input_image_name + '_enet_gt' + '.' + input_path_ext

        cv2.imwrite(out_path_im, prediction_rgb)
        # cv2.imwrite(out_path_gt, prediction) #  label images, where each pixel has an ID that represents the class







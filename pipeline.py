import os
import cv2
import caffe
import numpy as np
from math import floor
from functools import partial
from matplotlib import pyplot as plt


class LandmarksPipeline:

    def __init__(self, model_type):

        point_counts = {"full": 8, "upper": 6, "lower": 4}
        try:
            self.num_points = point_counts[model_type]
        except KeyError:
            raise Exception("Undefined model type.")

        models_folder = './models/FLD_{}'.format(model_type)

        model_stage1 = os.path.join(models_folder, 'stage1.prototxt')
        weights_stage1 = os.path.join(models_folder, 'stage1.caffemodel')
        model_stage2 = os.path.join(models_folder, 'cascade.prototxt')
        weights_stage2 = os.path.join(models_folder, 'stage2.caffemodel')
        model_stage3 = os.path.join(models_folder, 'cascade.prototxt')
        weights_stage3_easy = os.path.join(
                models_folder, 'stage3_easy.caffemodel')
        weights_stage3_hard = os.path.join(
                models_folder, 'stage3_hard.caffemodel')

        self.net_stage1 = caffe.Net(model_stage1, weights_stage1, caffe.TEST)
        self.net_stage2 = caffe.Net(model_stage2, weights_stage2, caffe.TEST)
        self.net_stage3_easy = caffe.Net(
                model_stage3, weights_stage3_easy, caffe.TEST)
        self.net_stage3_hard = caffe.Net(
                model_stage3, weights_stage3_hard, caffe.TEST)
    
    def get_original_coords(self, offset, scale, pt):
        pt = ((pt[0] + 0.5) * 224 - offset[1][0], (pt[1] + 0.5) * 224 - offset[0][0])
        pt = (pt[0] / scale, pt[1] / scale)
        return pt

    def preprocess(self, img):

        height, width = img.shape[:2]
        scale = 224 / max((height, width))
        s1, s2 = round(width * scale), round(height * scale)

        resized = cv2.resize(img, (s1, s2))
        pad = [a - b if a > b else 0 for a, b in zip([224, 224], resized.shape[:2])]

        offset = [
                (floor(pad[0] / 2), pad[0] - floor(pad[0] / 2)),
                (floor(pad[1] / 2), pad[1] - floor(pad[1] / 2)),
                ]

        padded = cv2.copyMakeBorder(resized, offset[0][0], offset[0][1],
                offset[1][0], offset[1][1], cv2.BORDER_CONSTANT, value=(0, 0, 0))


        assert(padded.shape[0] == 224)
        assert(padded.shape[1] == 224)

#        pixel_means = np.reshape([102.9801, 115.9465, 122.7717], [1, 1, 3])
#        just subtract above means if normalize doesn't work

        self.get_orig_coords = partial(self.get_original_coords, offset, scale)

        return padded, self.get_orig_coords


    def forward_pass(self, im_input):

        visibility_case = ['Visible', 'Occlude', 'Inexistent']

        # Pipeline stage 1
        self.net_stage1.blobs['data'].reshape(*im_input.shape)
        self.net_stage1.blobs['data'].data[...] = im_input

        res_stage1 = self.net_stage1.forward()
        print(res_stage1)
        landmarks_stage1 = res_stage1['fc8'][0][:self.num_points*2]
        pts_orig = landmarks_stage1.reshape((self.num_points, 2))
        visibility_vecs = res_stage1['fc8'][0][
                self.num_points * 2:].reshape((self.num_points, 3))
        visibilities_stage1 = [visibility_case[list(v).index(max(v))]
                for v in visibility_vecs]
        prediction_stage1 = {'landmarks':[self.get_orig_coords(pt) for pt in pts_orig],
                'visibilities':visibilities_stage1}

        print("Predictions 1:\n")
        print(prediction_stage1)

        # Pipeline stage 2
        self.net_stage2.blobs['data'].reshape(*im_input.shape)
        self.net_stage2.blobs['data'].data[...] = im_input
        self.net_stage2.blobs['prediction'].reshape(1, *landmarks_stage1.shape)
        self.net_stage2.blobs['prediction'].data[...] = np.array(landmarks_stage1)

        res_stage2 = self.net_stage2.forward()
        landmarks_stage2 = np.array([a - (b/5) for a, b in zip(
            landmarks_stage1, 
            res_stage2['fc8'][0][:self.num_points*2])])
        pts = landmarks_stage2.reshape((self.num_points, 2))
        visibility_vecs = res_stage2['fc8'][0][self.num_points * 2:
                ].reshape((self.num_points, 3))
        visibilities_stage2 = [visibility_case[list(v).index(max(v))]
                for v in visibility_vecs]
        prediction_stage2 = {'landmarks':[self.get_orig_coords(pt) for pt in pts],
                'visibilities':visibilities_stage2}

        print("Predictions 2:\n")
        print(prediction_stage2)

        # Pipeline stage 3
        self.net_stage3_easy.blobs['data'].reshape(*im_input.shape)
        self.net_stage3_easy.blobs['data'].data[...] = im_input
        self.net_stage3_easy.blobs['prediction'].reshape(1, *landmarks_stage2.shape)
        self.net_stage3_easy.blobs['prediction'].data[...] = np.array(landmarks_stage2)
        self.net_stage3_hard.blobs['data'].reshape(*im_input.shape)
        self.net_stage3_hard.blobs['data'].data[...] = im_input
        self.net_stage3_hard.blobs['prediction'].reshape(1, *landmarks_stage2.shape)
        self.net_stage3_hard.blobs['prediction'].data[...] = np.array(landmarks_stage2)


        res_stage3_easy = self.net_stage3_easy.forward()
        res_stage3_hard = self.net_stage3_hard.forward()
        landmarks_stage3 = np.array([a - (b/5 + c/5)/2 for a, b, c in zip(
            landmarks_stage2, 
            res_stage3_easy['fc8'][0][:self.num_points*2],
            res_stage3_hard['fc8'][0][:self.num_points*2]
            )])
        pts = landmarks_stage3.reshape((self.num_points, 2))
        visibility_vecs = res_stage3_easy['fc8'][0][self.num_points * 2:
                ].reshape((self.num_points, 3))
        visibilities_stage3 = [visibility_case[list(v).index(max(v))]
                for v in visibility_vecs]
        prediction_stage3 = {'landmarks':[self.get_orig_coords(pt) for pt in pts],
                'visibilities':visibilities_stage3}

        print("Predictions 3:\n")
        print(prediction_stage3)

        return pts, pts_orig

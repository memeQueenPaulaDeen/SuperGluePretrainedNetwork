from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np

from .models.matching import Matching
from .models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)

#idk wtf this is about
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class SuperGlueMatcher:

    def __init__(self):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Running inference on device \"{}\"'.format(self.device))
        self.config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': -1  # keep all
            },
            'superglue': {
                'weights': 'outdoor',
                'sinkhorn_iterations': 20,
                'match_threshold': .2,
            }
        }
        matching = Matching(self.config).eval().to(self.device)
        self.MatchModel = matching


    def matchImgs(self,img1,img2,isAffine):

        keys = ['keypoints', 'scores', 'descriptors']



        frame_tensor = frame2tensor(img1, self.device)
        last_data = self.MatchModel.superpoint({'image': frame_tensor})
        last_data = {k + '0': last_data[k] for k in keys}
        last_data['image0'] = frame_tensor


        frame_tensor = frame2tensor(img2, self.device)
        pred = self.MatchModel({**last_data, 'image1': frame_tensor})
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()

        valid = matches > -1

        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        if isAffine:
            homography, mask = cv2.estimateAffinePartial2D(mkpts1, mkpts0, None, cv2.RANSAC, 15)
            homography = np.r_[homography, np.array([0, 0, 1]).reshape(1, 3)]
        else:
            homography, mask = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5)
        return homography, mask





# if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #     description='SuperGlue demo',
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument(
    #     '--input', type=str, default='0',
    #     help='ID of a USB webcam, URL of an IP camera, '
    #          'or path to an image directory or movie file')
    # parser.add_argument(
    #     '--output_dir', type=str, default=None,
    #     help='Directory where to write output frames (If None, no output)')
    #
    # parser.add_argument(
    #     '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
    #     help='Glob if a directory of images is specified')
    # parser.add_argument(
    #     '--skip', type=int, default=1,
    #     help='Images to skip if input is a movie or directory')
    # parser.add_argument(
    #     '--max_length', type=int, default=1000000,
    #     help='Maximum length if input is a movie or directory')
    # parser.add_argument(
    #     '--resize', type=int, nargs='+', default=[640, 480],
    #     help='Resize the input image before running inference. If two numbers, '
    #          'resize to the exact dimensions, if one number, resize the max '
    #          'dimension, if -1, do not resize')
    #
    # parser.add_argument(
    #     '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
    #     help='SuperGlue weights')
    # parser.add_argument(
    #     '--max_keypoints', type=int, default=-1,
    #     help='Maximum number of keypoints detected by Superpoint'
    #          ' (\'-1\' keeps all keypoints)')
    # parser.add_argument(
    #     '--keypoint_threshold', type=float, default=0.005,
    #     help='SuperPoint keypoint detector confidence threshold')
    # parser.add_argument(
    #     '--nms_radius', type=int, default=4,
    #     help='SuperPoint Non Maximum Suppression (NMS) radius'
    #     ' (Must be positive)')
    # parser.add_argument(
    #     '--sinkhorn_iterations', type=int, default=20,
    #     help='Number of Sinkhorn iterations performed by SuperGlue')
    # parser.add_argument(
    #     '--match_threshold', type=float, default=0.2,
    #     help='SuperGlue match threshold')
    #
    # parser.add_argument(
    #     '--show_keypoints', action='store_true',
    #     help='Show the detected keypoints')
    # parser.add_argument(
    #     '--no_display', action='store_true',
    #     help='Do not display images to screen. Useful if running remotely')
    # parser.add_argument(
    #     '--force_cpu', action='store_true',
    #     help='Force pytorch to run in CPU mode.')







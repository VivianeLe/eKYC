import os
import cv2
import numpy as np
import onnxruntime

from .yolov8.inference import inference_yolov8
from .utils.dewarp_utils import polygon_from_corners, increase_border, distance
from lib.config.settings import *

PADDING_SIZE = 8   # padding around card

class CardDetection(object):
    def __init__(self, weight_path=ONNX_CARD_DETECTION_MODEL_PATH, device='cpu'):
        '''

        @param weight_path:
        @param device: 'cpu' or '0' or '0,1,2,3'
        @param onnx:
        '''
        self.card_model = self.__load_onnx_model(weight_path=weight_path)


    def __load_onnx_model(self, weight_path):
        provider = os.getenv('PROVIDER', 'CPUExecutionProvider')
        model = onnxruntime.InferenceSession(weight_path, providers=[provider])
        return model


    def detect(self, img):
        '''
        @param im: cv2 image: BGR mode
        @return:

        '''
        corners = self.detect_corners(img)
        card_dewarped = self.dewarp_image(img, corners)
        if card_dewarped is None:
            raise Exception("Cannot detect card from image!!!")
        return card_dewarped

    def detect_corners(self, img):

        if img is None:
            return []
        target = inference_yolov8(img, self.card_model)
        if target is None:
            return []
            
        pts = polygon_from_corners(target)
        if pts is None:
            return []
        else:
            pts = pts.astype(int)
            corners = increase_border(pts, PADDING_SIZE)
            corners = [(int(p[0]), int(p[1])) for p in corners]
            return corners

    def dewarp_image(self, img, corners):

        if img is None:
            return None
        if len(corners) != 4:
            return None
        target_w = int(max(distance(corners[0], corners[1]), distance(corners[2], corners[3])))
        target_h = int(max(distance(corners[0], corners[3]), distance(corners[1], corners[2])))
        target_corners = [[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]]

        pts1 = np.float32(corners)
        pts2 = np.float32(target_corners)
        transform_matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_dewarped = cv2.warpPerspective(img, transform_matrix, (target_w, target_h))

        return img_dewarped

from PIL import Image
import os
import onnxruntime
import cv2

from .yolov5.inference import inference_yolo
from .utils.line_detect_utils import polygon_from_corners, increase_size_box, crop_img_from_bbox
from lib.config.settings import ONNX_LINE_DETECTION_MODEL_PATH


IMG_SIZE = 640
CONF_THRES = 0.3
IOU_THRES = 0.5
MIN_PADDING_SIZE = 1
h_extend_size = 0.06
w_extend_size = 0.04


class LineDetection(object):
    def __init__(self, weight_path=ONNX_LINE_DETECTION_MODEL_PATH, device='cpu'):
        '''

        @param weight_path:
        @param device: 'cpu' or '0' or '0,1,2,3'
        @param onnx:
        '''
        self.line_model= self.__load_onnx_model(weight_path=weight_path)


    def __load_onnx_model(self, weight_path):
        provider = os.getenv('PROVIDER', 'CPUExecutionProvider')
        model = onnxruntime.InferenceSession(weight_path, providers=[provider])
        return model

    def detect(self, img):
        boxes_dict = self.detect_lines(img)
        img_drawed = self.draw_boxes(img, boxes_dict)
        line_img_dict = self.crop_lines(img, bbox_dict=boxes_dict)
        return line_img_dict, img_drawed

    def detect_lines(self, img):

        if img is None:
            return []
        info = {}
        list_dict = ['address_line_1', 'address_line_2', 'birthday', 'hometown_line_1', 'hometown_line_2', 'id', 'name',
                     'nation', 'sex', 'passport']
        target = inference_yolo(img, model=self.line_model, img_size=IMG_SIZE, conf_thres=CONF_THRES, iou_thres=IOU_THRES)
        if target is None:
            raise Exception("Cannot detect line from card")

        boxes = polygon_from_corners(target)
        if boxes is not None:
            for i, coord in enumerate(boxes):
                info[list_dict[i]] = coord
        else:
            return None
        return info

    def crop_lines(self, img, bbox_dict):
        '''

        @param img: cv2 image: BGR
        @return: list PIL image
        '''

        # Get boxes with score larger threshold
        if bbox_dict is None:
            return None


        list_label = ['address_line_1', 'address_line_2', 'birthday', 'hometown_line_1', 'hometown_line_2', 'id',
                      'name', 'nation', 'sex', 'passport']
        # processing
        for inf in list_label:
            bbox = bbox_dict[inf]
            if len(bbox) == 4:
                if bbox[3] > MIN_PADDING_SIZE:
                    bbox_dict[inf] = increase_size_box(bbox, img_size=img.shape, h_extend_size=h_extend_size, w_extend_size=w_extend_size)

        # Crop line
        line_img_dict = {}
        img_for_crop = Image.fromarray(img)
        for idx, label in enumerate(list_label):
            if len(bbox_dict[label]) == 4:
                bbox = bbox_dict[label]
                line_img_dict[label] = crop_img_from_bbox(img_for_crop, bbox)
        return line_img_dict

    def draw_boxes(self, img, boxes_dict, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes on an image.
        
        Args:
            image (np.array): The image on which to draw the boxes.
            boxes_dict (dict): A dictionary where keys are labels and values are bounding box coordinates.
            color (tuple): The color of the boxes. Default is green.
            thickness (int): The thickness of the box lines. Default is 2.
            
        Returns:
            The image with the bounding boxes drawn on it.
        """
        for label, box in boxes_dict.items():
            if box.size > 0:  # only draw box if array is not empty
                x1, y1, x2, y2 = box.astype(int)  # convert to int
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return img
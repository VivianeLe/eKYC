import time
import requests
import numpy as np
import cv2
from lib.card_detection.card_detect import CardDetection
from lib.line_detection.line_detect import LineDetection
from lib.ocr.ocr_recognition import OCR
from requests.exceptions import HTTPError

class InferencePipeline:

    def __init__(self,
                 device='cpu',
                ):
        '''

        @param device: 'cpu' or '0' or '0,1,2'
        '''

        self.card_detect_module = CardDetection(device=device)
        self.line_detect_module = LineDetection(device=device)
        # self.text_cls = PaddleTextClassifier(
        #     config_path=cls_config_path,
        #     use_gpu=cls_use_gpu)
        # self.detect_model = PaddleTextDetector(
        #     config_path=det_config_path,
        #     use_gpu=det_use_gpu) 
        self.ocr_module = OCR(device=device)
        self.result_keys = ['id', 'name', 'birthday', 'sex', 'nation', 'hometown', 'address']
        self.mapping = {'name': 'name', 'id': 'id', 'birthday': 'birthday', 'sex': 'sex',
                        'nation': 'nation', 'address': ['address_line_1', 'address_line_2'],
                        'hometown': ['hometown_line_1', 'hometown_line_2']}

    def run(self, image, format=True):
        '''

        @param image_url: cv2 image: BGR mode (important !!!)
        @param format:
        @return:
        '''
        try:
            start_time = time.time()
            if image is str:
                image = cv2.imread(image)
            else:
                image = image
        except:
            print('Can not load image')
            return 0

        try:
            start_ai_time = time.time()
            try:
                card_dewarped = self.card_detect_module.detect(image)
            except:
                card_dewarped = image

            line_img_dict, img_drawed = self.line_detect_module.detect(card_dewarped)
            result_ocr = self.ocr_module.recognize(line_img_dict, post_processing=True)
            if format:
                result_ocr = self.format_result(result_ocr)
            print("AI inference time: {:.2f}".format((time.time() - start_ai_time)))
        except Exception as err:
            print(err)
            return {}, None
        else:
            return result_ocr, img_drawed

    def format_result(self, result_ocr):
        '''
        Convert result_ocr to formated result by mapping dict
        @param result_ocr:
        @return:
        '''
        result_formated = {}
        for key in self.result_keys:
            if key == 'address':
                fields = self.mapping[key]
                address = []
                for field in fields:
                    if field in result_ocr:
                        address.append(result_ocr[field][0])
                result_formated[key] = " ".join(address)
            elif key == 'hometown':
                fields = self.mapping[key]
                address = []
                for field in fields:
                    if field in result_ocr:
                        address.append(result_ocr[field][0])
                result_formated[key] = " ".join(address)
            else:
                try:
                    result_formated[key] = result_ocr[self.mapping[key]][0]
                except:
                    result_formated[key] = ""
        return result_formated

if __name__=='__main__':
    img_url = ''
    inference_pipeline = InferencePipeline(device='cpu')
    result = inference_pipeline.run(str(img_url))
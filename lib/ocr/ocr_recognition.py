from .vietocr.tool.predictor import Predictor
import os
import yaml
from .utils.ocr_utils import post_processing_result
from lib.config.settings import VIETOCR_MODEL_PATH

REG_MODEL = os.path.join(os.path.dirname(__file__), 'vietocr/config/config.yml')
BASE_MODEL = os.path.join(os.path.dirname(__file__), 'vietocr/config/base.yml')

class OCR(object):
    
    def __init__(self, weight_path=VIETOCR_MODEL_PATH, device='cpu'):
        '''

        @param weight_path:
        @param device: cpu or cuda
        '''
        if device != 'cpu':
            device = 'cuda'
        self.ocr_model = self.__load_model(weight_path=weight_path, device=device)

    def __load_model(self, weight_path, device='cpu'):
        with open(BASE_MODEL, encoding="utf8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        with open(REG_MODEL, encoding="utf8") as file:
            config_overwrite = yaml.load(file, Loader=yaml.FullLoader)

        config.update(config_overwrite)

        config['weights'] = weight_path
        config['cnn']['pretrained'] = False
        config['device'] = device
        config['predictor']['beamsearch'] = False
        ocr_model = Predictor(config)
        return ocr_model

    def recognize(self, line_img_dict, post_processing=True):
        '''

        @param result_line_img:
        @return:

            {'field_1': ['str1'], 'field_2: ['str2'],...}
        '''

        result_ocr = {}
        for field, img in line_img_dict.items():
            result_ocr[field] = []

            res_str = self.ocr_model.predict(img)
            result_ocr[field].append(res_str)

        if post_processing:
            try:
                result_ocr = post_processing_result(result_dict=result_ocr, norm_sex=True, norm_nation=True, norm_name=True)
            except:
                return result_ocr

        return result_ocr


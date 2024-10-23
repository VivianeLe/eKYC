import onnxruntime as ort
from PIL import Image
import cv2
import numpy as np

IMG_SIZE = 512
CONF_THRES = 0.3
IOU_THRES = 0.5

def inference_yolov8(image, model):
    """
        Handler of /detect POST endpoint
        Receives uploaded file with a name "image_file", passes it
        through YOLOv8 object detection network and returns and array
        of bounding boxes.
        :return: a JSON array of objects bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
    """
    boxes = detect_objects_on_image(image, model)
    return boxes


def detect_objects_on_image(image, model):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param image: Input image
    :return: Array of bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
    """
    input, img_width, img_height = prepare_input(image)
    output = run_model(input, model)
    return process_output(output, img_width, img_height)


def prepare_input(img):
    """
    Function used to convert input image to tensor,
    required as an input to YOLOv8 object detection
    network.
    :param image: image
    :return: Numpy array in a shape (3,width,height) where 3 is number of color channels
    """
    img_width, img_height, _ = img.shape
    img = cv2.resize(img,(IMG_SIZE, IMG_SIZE))
    input = np.array(img) / 255.0
    input = input.transpose(2, 0, 1)
    input = input.reshape(1, 3, IMG_SIZE, IMG_SIZE)
    return input.astype(np.float32), img_width, img_height


def run_model(input, model):
    """
    Function used to pass provided input tensor to
    YOLOv8 neural network and return result
    :param input: Numpy array in a shape (3,width,height)
    :return: Raw output of YOLOv8 network as an array of shape (1,4,8400)
    """
    outputs = model.run(["output0"], {"images":input})
    return outputs[0]


def process_output(output, img_width, img_height):
    """
    Function used to convert RAW output from YOLOv8 to an array
    of detected objects. Each object contain the bounding box of
    this object, the type of object and the probability
    :param output: Raw output of YOLOv8 network which is an array of shape (1,84,8400)
    :param img_width: The width of original image
    :param img_height: The height of original image
    :return: Array of detected objects in a format [[x1,y1,x2,y2,object_type,probability],..]
    """
    output = output[0].astype(float)
    output = output.transpose()

    boxes = []
    for row in output:
        prob = row[4:].max()
        if prob < CONF_THRES:
            continue
        class_id = row[4:].argmax()
        label = class_id
        xc, yc, w, h = row[:4]
        x1 = (xc - w/2) / IMG_SIZE * img_width
        y1 = (yc - h/2) / IMG_SIZE * img_height
        x2 = (xc + w/2) / IMG_SIZE * img_width
        y2 = (yc + h/2) / IMG_SIZE * img_height
        boxes.append([label, x1, y1, x2, y2, prob])

    boxes.sort(key=lambda x: x[5], reverse=True)
    result = []
    while len(boxes) > 0:
        result.append(boxes[0])
        boxes = [box for box in boxes if iou(box, boxes[0]) < IOU_THRES]

    return np.array(result)


def iou(box1,box2):
    """
    Function calculates "Intersection-over-union" coefficient for specified two boxes
    https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
    :param box1: First box in format: [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format: [x1,y1,x2,y2,object_class,probability]
    :return: Intersection over union ratio as a float number
    """
    return intersection(box1,box2)/union(box1,box2)


def union(box1,box2):
    """
    Function calculates union area of two boxes
    :param box1: First box in format [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format [x1,y1,x2,y2,object_class,probability]
    :return: Area of the boxes union as a float number
    """
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)


def intersection(box1,box2):
    """
    Function calculates intersection area of two boxes
    :param box1: First box in format [x1,y1,x2,y2,object_class,probability]
    :param box2: Second box in format [x1,y1,x2,y2,object_class,probability]
    :return: Area of intersection of the boxes as a float number
    """
    box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
    box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
    x1 = max(box1_x1,box2_x1)
    y1 = max(box1_y1,box2_y1)
    x2 = min(box1_x2,box2_x2)
    y2 = min(box1_y2,box2_y2)
    return (x2-x1)*(y2-y1)



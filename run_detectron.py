import os
import cv2
import json
import random
import itertools
import numpy as np
import pickle
import time
# from detectron2.engine import DefaultPredictor
# #from detectron2.detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.config import get_cfg
# #from detectron2.utils.visualizer import Visualizer, ColorMode
# #from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
# from detectron2.structures import BoxMode

# import json

# cfg = get_cfg()
# cfg.MODEL.DEVICE='cpu'
# '''Load Faster RCNN
# cfg.MODEL.WEIGHTS = "./model/faster_rcnn_R_101_FPN_3x_model/model_final.pth"'''
# # --- #
# # Load Mask RCNN configuration
# cfg.merge_from_file("detectron/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
# cfg.MODEL.WEIGHTS = "model/mask_rcnn_R_50_FPN_3x_model/model_final.pth"
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95 # Correct class result must be more than 95%
# predictor = DefaultPredictor(cfg)


from detectron2.engine import DefaultPredictor
#from detectron2.detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
#from detectron2.utils.visualizer import Visualizer, ColorMode
#from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import imutils
import json

cfg = get_cfg()
cfg.MODEL.DEVICE='cpu'

# cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
# cfg.MODEL.WEIGHTS = "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Correct class result must be more than 50%
predictor = DefaultPredictor(cfg)

def _create_text_labels(classes, scores, class_names):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 0:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    return labels

def object_detect(image):
    predictions = predictor(image)
    boxes = predictions["instances"].pred_boxes 
    scores = predictions["instances"].scores 
    classes = predictions["instances"].pred_classes 
    # labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))


    return predictions, boxes, scores, classes

if __name__ == '__main__':
    image = cv2.imread("demo.png")
    image = imutils.resize(image, width=400)
    outputs, boxes, scores, classes = object_detect(image)
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("image", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
# coding: utf-8

#History
# 4/13: first imp ver.


import os
import numpy as np
import torch
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import postprocess as yolox_postprocess
from yolox.exp import Exp as MyExp
import glob

class yolox_s_Exp(MyExp):
    def __init__(self):
        super(yolox_s_Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 1
        self.exp_name = 'yolox_s'

class Predictor(object):
    _defaults = {
        'model_path': 'pretrain/yolox_s.pth',
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"        
    
    def __init__(self, conf=0.25, device='gpu', **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)        
        exp = yolox_s_Exp()
        exp.test_conf = conf
        exp.nmsthre = 0.46
        exp.test_size = (640, 640)
        model = exp.get_model()
        model.cuda();
        model.eval();
        ckpt = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        self.model = model
        self.cls_names = ('flower')
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = False
        self.preproc = ValTransform(legacy=False)

    def detect(self, img, person_only=False, bbox_area_thres=0):
        height, width = img.shape[:2]
        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == 'gpu':
            img = img.cuda()
            if self.fp16:
                img = img.half()
        with torch.no_grad():
            outputs = self.model(img)
            outputs = yolox_postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
        output = outputs[0]
        if output is None:
            return [],[],[],[]

        #keep person cls
        output = output.cpu().numpy()
        bboxes_tlbr = output[:, 0:4]
        bboxes_tlbr /= ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        boxes_tlwh = [self.to_tlwh(box) for box in bboxes_tlbr]
        return cls, scores, bboxes_tlbr, boxes_tlwh
    
    def parse_bbox(self, bboxs):
        area_list = np.array(list(map(lambda d: (d[2]-d[0])*(d[3]-d[1]), bboxs)))
        max_area = max(area_list)
        area_ratio_list = np.array([a/max_area for a in area_list])
        valid_idx = np.argwhere(area_ratio_list>0.2).flatten()
        bboxs = bboxs[valid_idx]  
        return bboxs, valid_idx
    
    def merge_bbox(self, bboxs):
        bbox = bboxs[:,0].min(), bboxs[:,1].min(), bboxs[:,2].max(), bboxs[:,3].max()        
        return bbox

    def to_tlbr(self, tlwh):
        tlbr = tlwh.copy()
        tlbr[2:] = tlbr[:2] + tlbr[2:]
        return tlbr

    def to_tlwh(self, tlbr):
        x1, y1, x2, y2 = tlbr
        w, h = x2-x1, y2-y1
        return x1, y1, w, h    
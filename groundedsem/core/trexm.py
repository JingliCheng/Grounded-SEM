import logging
import os
from pathlib import Path

import torch
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont


# groundedsem
from groundedsem import defaults
from segment_anything import build_sam, SamPredictor
from groundedsem.util.utils import load_model_hf, segment, draw_mask, download_file_if_not_exist
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
from groundedsem import logger

from .base import ModelBase


class TrexM(ModelBase):
    def __init__(self, sam_model=None, device=None, **kwargs):
        super().__init__(sam_model, device, **kwargs)
        

    def sam_segment(self, image_source, boxes):
        self.sam_predictor.set_image(image_source)
        boxes = torch.tensor(boxes)
        
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes.to(self.device), image_source.shape[:2])
        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
            )
        return masks.cpu()
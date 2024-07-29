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


class GSEM:
    def __init__(self, dino_model=None, sam_model=None, device=None, **kwargs):
        self.PACKAGE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
        # device
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        # DINO
        if not dino_model:
            logger.info("Use default DINO. Downloading...")
            self.dino_model = load_model_hf(
                defaults.CKPT_REPO_ID, 
                defaults.CKPT_FILENAME, 
                defaults.CKPT_CONFIG_FILENAME, 
                self.device
                )
            logger.info("DINO Done")
        else:
            self.dino_model = dino_model
        # SAMs
        if not sam_model:
            logger.info("Use default SAM. Downloading...")
            sam_local = self.PACKAGE_DIR.joinpath(defaults.SAM_CHECKPOINT)
            download_file_if_not_exist(defaults.SAM_URL, sam_local)
            logger.info("SAM Done")
            self.sam_model = build_sam(sam_local)
        else:
            self.sam_model = sam_model
        self.sam_predictor = SamPredictor(self.sam_model.to(self.device))
        # Others
        self.params = kwargs.get('params', {})


    # detect object using grounding DINO
    def dino_predict(self, image, text_prompt='particle', box_threshold = 0.1, text_threshold = 0.25):
        boxes, logits, phrases = predict(
            model= self.dino_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        return boxes, logits, phrases


    def detect(self, image_path, text_prompt='particle', box_threshold = 0.1, text_threshold = 0.25):
        image_source, image_transed = load_image(image_path)
        boxes, logits, phrases = self.dino_predict(
            image=image_transed, 
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        annotated_frame = annotate(image_source, boxes, logits, phrases)
        annotated_frame = annotated_frame[...,::-1] # BGR to RGB
        Image.fromarray(annotated_frame).show()
        return annotated_frame, boxes, logits, phrases


    def sam_segment(self, image_source, boxes, annotated_frame):
        segmented_frame_masks = segment(
            image_source, 
            self.sam_predictor, 
            boxes=boxes, 
            device=self.device
            )
        return segmented_frame_masks


    def dino_sam_segment(self, image_path, text_prompt='particle', box_threshold = 0.1, text_threshold = 0.25):
        # Load image
        image_source, image_transed = load_image(image_path)
        # DINO
        boxes, logits, phrases = self.dino_predict(
            image=image_transed, 
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        annotated_frame = annotate(image_source, boxes, logits, phrases)
        annotated_frame = annotated_frame[...,::-1] # BGR to RGB
        # SAM
        segmented_frame_masks = self.sam_segment(image_source, boxes, annotated_frame)
        for mask in segmented_frame_masks[1:]:
            annotated_frame = draw_mask(mask[0], annotated_frame)
        Image.fromarray(annotated_frame).show()
        return annotated_frame

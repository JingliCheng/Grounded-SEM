from typing import Any, Dict, List
import requests
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from groundingdino.util.utils import clean_state_dict
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from huggingface_hub import hf_hub_download


def load_config(repo_id, ckpt_config_filename):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
    args = SLConfig.fromfile(cache_config_file)
    return args

def load_checkpoint(repo_id, filename, device):
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    return checkpoint, cache_file

def build_and_load_model(args, checkpoint, device):
    # Build the model using the configuration settings
    model = build_model(args)
    args.device = device
    # Load the model state dictionary
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    return model, log

def load_model_hf(repo_id, filename, ckpt_config_filename, device):
    """
    from https://github.com/JingliCheng/Grounded-SEM/blob/main/grounded_sam_colab_demo.ipynb
    """
    args = load_config(repo_id, ckpt_config_filename)
    print(args)
    checkpoint, cache_file = load_checkpoint(repo_id, filename, device)
    model, log = build_and_load_model(args, checkpoint, device)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    return model


def segment(image, sam_model, boxes, device='cpu'):
    """
    from https://github.com/JingliCheng/Grounded-SEM/blob/main/grounded_sam_colab_demo.ipynb
    """
    sam_model.set_image(image)
    H, W, _ = image.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    
    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
    masks, _, _ = sam_model.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output = False,
        )
    return masks.cpu()


def draw_mask(mask, image, random_color=True):
    """
    from https://github.com/JingliCheng/Grounded-SEM/blob/main/grounded_sam_colab_demo.ipynb
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


def download_file(url: str, local_filename: str):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def download_file_if_not_exist(url: str, local_filename: str):
    if not os.path.exists(local_filename):
        download_file(url, local_filename)
    else:
        print(f"{local_filename} exists, skipping download.")


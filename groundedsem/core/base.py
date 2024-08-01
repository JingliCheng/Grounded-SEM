import os
from pathlib import Path

import torch

# groundedsem
from groundedsem import defaults
from segment_anything import build_sam, SamPredictor
from groundedsem.util.utils import download_file_if_not_exist
from groundedsem import logger


class ModelBase:
    def __init__(self, sam_model=None, device=None, **kwargs):
        self.PACKAGE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
        # device
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
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

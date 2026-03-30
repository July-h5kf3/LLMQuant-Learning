import torch
import torch.nn

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN, IGNORE_INDEX,
                         IMAGE_TOKEN_INDEX)
from .multimodel_encoder.
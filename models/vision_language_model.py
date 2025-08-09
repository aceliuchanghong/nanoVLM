import torch
import torch.nn as nn
from transformers import AutoModelForVision2Seq, AutoProcessor
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)
from models.config import VLMConfig


class VisionLanguageModel(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config

        # 加载视觉主干 (Vision Backbone)
        qwen_vl_model = AutoModelForVision2Seq.from_pretrained(
            config.vision_model_id, torch_dtype=torch.bfloat16
        )
        self.vision_backbone = qwen_vl_model.vision_encoder

        # 加载与视觉模型配套的图像处理器
        self.image_processor = AutoProcessor.from_pretrained(config.vision_model_id)

        # 参数冻结
        for param in self.vision_backbone.parameters():
            param.requires_grad = False

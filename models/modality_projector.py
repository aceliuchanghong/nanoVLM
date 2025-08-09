import torch
import torch.nn as nn

import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)
from models.config import VLMConfig


class ModalityProjection(nn.Module):
    """模态投影-模态的对齐,视觉维度==>语言维度"""

    def __init__(self, config: VLMConfig):
        super().__init__()
        self.use_pixel_shuffle = config.use_pixel_shuffle
        if self.use_pixel_shuffle:
            # TODO 之后补充
            pass

        self.linear = nn.Linear(config.vision_hidden_size, config.lang_hidden_size)

    def forward(self, vision_embeddings: torch.Tensor) -> torch.Tensor:
        # vision_embeddings 的形状: (batch_size, num_vision_tokens, vision_hidden_size)
        # 直接通过线性层进行投影
        projected_embeddings = self.linear(vision_embeddings)
        # 输出形状: (batch_size, num_vision_tokens, lang_hidden_size)
        return projected_embeddings

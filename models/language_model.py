import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)
from models.config import VLMConfig


class LanguageModel(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config

        self.lang_backbone = AutoModelForCausalLM.from_pretrained(
            config.lang_model_id, torch_dtype=torch.bfloat16
        )

        # 加载与语言模型配套的分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.lang_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # 设置pad_token

        for param in self.lang_backbone.parameters():
            param.requires_grad = False

        # 获取语言模型的词嵌入层，用于后续计算
        self.lang_embeddings = self.lang_backbone.get_input_embeddings()

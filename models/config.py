from dataclasses import dataclass


@dataclass
class VLMConfig:
    # 定义我们选择的模型ID
    vision_model_id: str = "/mnt/data/llch/Models/Qwen2.5-VL-3B-Instruct"
    lang_model_id: str = "/mnt/data/llch/Models/Qwen3-4B-Instruct-2507"

    # 视觉编码器的输出维度
    vision_hidden_size: int = 1280
    # 语言模型的输入维度
    lang_hidden_size: int = 2560

    # 语言模型的词汇表大小，可以从tokenizer动态获取
    vocab_size: int = 152064  # Qwen3-4B-Instruct-2507 的 vocab_size

    # 图像块相关配置 (可以保持nanoVLM默认或根据Qwen-VL调整)
    patch_size: int = 14  # Qwen2.5-VL-3B uses patch_size 14
    image_size: int = (
        448  # Qwen2.5-VL-3B supports dynamic resolution, 448 is a good default
    )

    # 模态投影层配置
    use_pixel_shuffle: bool = True
    pixel_shuffle_factor: int = 2  # 保持默认，将视觉token数量减少4倍


@dataclass
class TrainConfig:
    # 训练超参数
    batch_size: int = 8
    epochs: int = 5

    # 学习率设置 (采用差分学习率)
    projector_lr: float = 1e-3
    finetune_lr: float = 1e-5

    # 训练控制
    num_workers: int = 4
    device: str = "cuda"

    # 日志和保存
    log_wandb: bool = True
    checkpoint_path: str = "./checkpoints/"
    run_name: str = "docuvlm-qwen-run"

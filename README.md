# nanoVLM

```
llm: /mnt/data/llch/Models/Qwen3-4B-Instruct-2507
vlm: /mnt/data/llch/Models/Qwen2.5-VL-3B-Instruct
dataset: 
- /mnt/data/llch/my_lm_log/6_代码实现/nanovlm/the_cauldron
- /mnt/data/llch/Models/dataset/OCRFlux-bench-single/
```

```
+-----------------------------+
|           输入数据           |
|  图像 (Image) → 视觉编码器   |
|  文本 (Text) → 分词 + 嵌入   |
+------------+----------------+
             |
+------------v----------------+     +----------------------------+
|    视觉编码器 (ViT)          |     | 文本嵌入 (Token Embedding) |
| - 提取图像 patch embeddings  |---->|                           |
| - 输出: [B, N, D]            |     |                           |
+-----------------------------+     +------------+-------------+
                                                 |
                      +---------------------------v--------------------------+
                      |        Cross-Attention 模块 (跨模态交互)           |
                      | - Query 来自文本嵌入                               |
                      | - Key/Value 来自图像特征                           |
                      | - 输出融合后的上下文信息 [B, T, D]                  |
                      +---------------------------+--------------------------+
                                                  |
                      +---------------------------v--------------------------+
                      |         轻量级语言解码器                              |
                      | - 使用融合后的上下文作为输入                           |
                      | - 自回归生成文本                                      |
                      | - 支持训练和推理                                      |
                      +------------------------------------------------------+
                                                  |
                                          +---------v----------+
                                          |       损失函数        |
                                          | - 使用交叉熵损失      |
                                          | - 对比学习（可选）    |
                                          +----------------------+
```


```shell
# linux
source .venv/bin/activate

# 设置代理源
pip install requests -i https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple requests 
uv add requests -i https://pypi.tuna.tsinghua.edu.cn/simple
vi ~/.bashrc==>export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple==>source ~/.bashrc

hf download ChatDOC/OCRFlux-bench-single --repo-type dataset --local-dir /mnt/data/llch/Models/dataset/OCRFlux-bench-single
tar xzvf /mnt/data/llch/Models/dataset/OCRFlux-bench-single/pdfs.tar.gz -C /mnt/data/llch/Models/dataset/OCRFlux-bench-single/
```

### reference
- [拼接smvlm](https://docs.swanlab.cn/examples/qwen3_smolvlm_muxi.html)
- [数据-the_cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)
- [数据-OCRFlux](https://huggingface.co/datasets/ChatDOC/OCRFlux-bench-single)
- [Qwen3-SmVL](https://github.com/aceliuchanghong/Qwen3-SmVL)
- 
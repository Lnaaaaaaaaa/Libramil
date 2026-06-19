"""
Libra-MIL 文本编码器模块
========================
使用CONCH模型将文本描述编码为特征向量

CONCH简介:
    - CONCH (CONtrastive learning for Clinical Histopathology)
    - 微软开发的病理学专用视觉-语言基础模型
    - 论文: Lu et al. 2024, Nature Medicine
    - 预训练于大规模病理图像-文本对

为什么使用CONCH:
    1. 针对病理学领域优化，理解医学术语
    2. 视觉和文本特征在同一语义空间对齐
    3. 支持零样本分类和少样本学习
"""

import torch
import torch.nn as nn
from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize


class TextEncoder(nn.Module):
    """
    文本编码器类

    功能:
        将病理学文本描述编码为512维特征向量

    模型架构:
        - 基于ViT-B-16 (Vision Transformer - Base, patch size 16)
        - 输出维度: 512 (与视觉特征维度对齐)

    使用示例:
        encoder = TextEncoder(output_dim=512)
        texts = ["clear cell carcinoma", "papillary renal cell carcinoma"]
        features = encoder(texts)  # shape: (2, 512)
    """

    def __init__(self, output_dim=512, weights_path="conch/pytorch_model.bin"):
        """
        初始化文本编码器

        参数:
            output_dim: 输出特征维度，默认512
            weights_path: CONCH模型权重路径
        """
        super().__init__()

        # 获取CONCH的分词器
        # 用于将文本转换为模型输入的token序列
        self.tokenizer = get_tokenizer()

        # 加载预训练的CONCH模型
        # 'conch_ViT-B-16' 表示使用ViT-Base架构，patch大小为16x16
        self.base_model, _ = create_model_from_pretrained(
            'conch_ViT-B-16',
            weights_path
        )

    def forward(self, texts):
        """
        编码文本

        参数:
            texts: 文本列表，例如 ["clear cell carcinoma", "papillary carcinoma"]

        返回:
            text_feats: 文本特征张量，shape [N, output_dim]
                        N是文本数量，output_dim是特征维度(512)

        处理流程:
            1. tokenize: 将文本转为token ID序列
            2. encode_text: 通过CONCH的文本编码器提取特征
            3. 返回特征向量（已归一化到单位球面）

        注意:
            - 使用 torch.no_grad() 禁用梯度计算
            - 因为CONCH是冻结的，不参与训练
        """
        # 文本分词：将字符串转换为模型可处理的token序列
        # 例如: "clear cell carcinoma" -> [101, 2345, 6789, ...]
        tokenized = tokenize(texts=texts, tokenizer=self.tokenizer)

        # 使用CONCH编码文本（不计算梯度，模型冻结）
        with torch.no_grad():
            text_feats = self.base_model.encode_text(tokenized)

        return text_feats

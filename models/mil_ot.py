"""
Libra-MIL 核心模型模块
======================
实现基于最优传输的多模态原型学习框架

核心创新点:
    1. 双原型学习 (Dual-Prototype Learning)
       - 视觉原型: 可学习参数，捕捉视觉特征模式
       - 文本原型: 来自LLM生成的病理描述，提供语义先验

    2. 立体式最优传输 (Stereoscopic Optimal Transport, SOT)
       - 将视觉和文本相似度矩阵视为两个"视角"
       - 通过最优传输找到两个视角之间的最佳对齐
       - 实现双向跨模态信息融合

    3. 查询式聚合 (Query-based Aggregation)
       - 使用bag级别的文本先验作为查询
       - 通过交叉注意力机制聚合实例信息

模型架构图:
    ┌─────────────────────────────────────────────────────────────┐
    │                      Input: V_patch (B, N, D)               │
    │                           WSIs的patch特征                    │
    └─────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   投影层 (proj_v)                            │
    │              将特征映射到原型空间                             │
    └─────────────────────────────────────────────────────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                       ▼
    ┌─────────────────────┐               ┌─────────────────────┐
    │   视觉分支          │               │   文本分支          │
    │                     │               │                     │
    │  P_vis (可学习)     │               │  prompt_inst        │
    │  视觉原型           │               │  (来自LLM)          │
    └─────────────────────┘               └─────────────────────┘
              │                                       │
              ▼                                       ▼
    ┌─────────────────────┐               ┌─────────────────────┐
    │  计算相似度         │               │  计算相似度         │
    │  attn_vis           │               │  attn_struct        │
    └─────────────────────┘               └─────────────────────┘
              │                                       │
              └───────────────────┬───────────────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │   Sinkhorn最优传输 (SOT)    │
                    │   找到最佳对齐方案           │
                    └─────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   融合注意力 attn_fused     │
                    └─────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   特征加权 patch_fused      │
                    └─────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   交叉注意力聚合            │
                    │   Query: prompt_bag         │
                    │   Key/Value: patch_fused    │
                    └─────────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   分类头 → logits           │
                    └─────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 辅助函数
# ============================================================================

def pairwise_cosine_distance(x, y):
    """
    计算两组向量之间的成对余弦距离

    参数:
        x: (K1, D) 第一组向量
        y: (K2, D) 第二组向量

    返回:
        distance: (K1, K2) 余弦距离矩阵
                  distance[i,j] = 1 - cos(x[i], y[j])

    说明:
        余弦距离 = 1 - 余弦相似度
        - 当两向量方向相同时，距离=0
        - 当两向量方向相反时，距离=2
    """
    # L2归一化
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    # 计算余弦相似度矩阵，然后取反
    return 1.0 - torch.matmul(x, y.t())


def sinkhorn_ot(mu, nu, cost, epsilon=0.05, n_iters=20):
    """
    Sinkhorn最优传输算法

    功能:
        在两个分布之间找到最优传输方案（耦合矩阵）

    参数:
        mu: (B, N, K1) 源分布，通常是实例到视觉原型的注意力分布
        nu: (B, N, K2) 目标分布，通常是实例到文本原型的注意力分布
        cost: (K1, K2) 传输代价矩阵，通常是原型间的余弦距离
        epsilon: 熵正则化系数，控制传输方案的平滑程度
        n_iters: Sinkhorn迭代次数

    返回:
        T: (B, N, K1, K2) 最优传输方案（耦合矩阵）
           T[b,n,i,j] 表示第b个样本第n个实例中，视觉原型i到文本原型j的传输量

    算法原理:
        最优传输问题: min <T, C> - ε*H(T)
        其中 <T,C> 是传输代价，H(T) 是熵正则项

        Sinkhorn算法通过交替归一化求解:
        1. u = mu / (K @ v)
        2. v = nu / (K^T @ u)
        其中 K = exp(-C/ε) 是核矩阵

    直观理解:
        想象有一批货物(mu)要运送到目的地(nu)
        代价矩阵C决定运输成本
        Sinkhorn算法找到成本最低的运输方案T
    """
    B, N, K1 = mu.shape
    _, _, K2 = nu.shape

    # 扩展代价矩阵到batch维度: (K1, K2) -> (B, N, K1, K2)
    cost = cost.unsqueeze(0).unsqueeze(0).expand(B, N, K1, K2)

    # 计算核矩阵 K = exp(-C/ε)
    # 代价越小，核值越大，传输概率越高
    K_mat = torch.exp(-cost / epsilon)

    # 初始化对偶变量 u, v
    u = torch.ones_like(mu) / K1  # 均匀分布
    v = torch.ones_like(nu) / K2

    # Sinkhorn迭代
    for _ in range(n_iters):
        # 行归一化：使 T.sum(dim=-1) = mu
        u = mu / (torch.einsum("bnij,bnj->bni", K_mat, v) + 1e-8)
        # 列归一化：使 T.sum(dim=-2) = nu
        v = nu / (torch.einsum("bnij,bni->bnj", K_mat, u) + 1e-8)

    # 计算最终的传输矩阵
    T = K_mat * u.unsqueeze(-1) * v.unsqueeze(-2)

    return T

# ============================================================================
# 主模型类
# ============================================================================

class MIL_MultiPrompt_OTFusion(nn.Module):
    """
    Libra-MIL主模型

    核心组件:
        1. 视觉原型 (proto_vis): 可学习的语义锚点，捕捉视觉特征模式
        2. 文本原型 (prompt_inst): 来自LLM的病理描述，提供语义先验
        3. 最优传输融合: 实现视觉和文本原型的双向对齐
        4. 交叉注意力聚合: 使用bag级别查询聚合实例信息

    参数说明:
        dim: 特征维度 (默认512)
        num_struct_prompts: 实例级文本原型数量
        num_vis_prototypes: 视觉原型数量
        num_classes: 分类类别数
        T_struct_llm: 实例级文本原型特征 (来自TextEncoder)
        T_bag_llm: bag级文本原型特征 (来自TextEncoder)
    """

    def __init__(
            self,
            dim,
            num_struct_prompts,
            num_vis_prototypes,
            num_classes,
            T_struct_llm,
            T_bag_llm,
            pooling_type="attention",
            use_proj=True,
            ot_epsilon=0.05,
            ot_iter=20,
            loss_weights=None,
            ablation_setting=None,
            num_heads=8
        ):
        """
        初始化模型

        参数:
            dim: 特征维度，需与CONCH输出维度一致 (512)
            num_struct_prompts: 实例级文本原型数量 (由LLM生成的描述数量)
            num_vis_prototypes: 视觉原型数量 (可调超参数，论文建议6)
            num_classes: 分类类别数
            T_struct_llm: 预计算的实例级文本原型特征
            T_bag_llm: 预计算的bag级文本原型特征
            pooling_type: 池化类型 (attention/gated_attention/mean)
            use_proj: 是否使用投影层
            ot_epsilon: 最优传输的熵正则化系数
            ot_iter: Sinkhorn迭代次数
            num_heads: 注意力头数
        """
        super().__init__()

        # ========== 保存超参数 ==========
        self.dim = dim
        self.K1 = num_struct_prompts      # 文本原型数量
        self.K2 = num_vis_prototypes      # 视觉原型数量
        self.C = num_classes              # 类别数
        self.use_proj = use_proj
        self.ot_epsilon = ot_epsilon
        self.ot_iter = ot_iter
        

        # ========== 注册文本原型为buffer ==========
        # buffer: 不参与梯度更新的参数，但会随模型移动到GPU
        # 这些是LLM生成的病理描述，经过TextEncoder编码后的特征
        self.register_buffer("T_struct_llm", T_struct_llm)  # 实例级文本原型
        self.register_buffer("T_bag_llm", T_bag_llm)        # bag级文本原型

        self.ablation_setting = ablation_setting

        # ========== 可学习原型 ==========
        # 视觉原型：随机初始化，通过训练学习有意义的视觉模式
        # 论文图4展示了学习到的原型对应具体组织学特征
        self.proto_vis = nn.Parameter(torch.randn(self.K2, dim))

        # 文本原型：从LLM生成的描述初始化，可微调
        self.prompt_inst = nn.Parameter(self.T_struct_llm.clone())  # 实例级
        self.prompt_bag = nn.Parameter(self.T_bag_llm.clone())      # bag级

        # ========== 投影层 ==========
        # 将特征映射到统一的原型空间
        if use_proj:
            self.proj_v = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
            )
            self.proj_llm = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
            )
        else:
            self.proj_v = nn.Identity()
            self.proj_llm = nn.Identity()

        # ========== 温度参数 ==========
        # 控制softmax的锐度，可学习
        # 较小的温度 → 更尖锐的注意力分布
        # 较大的温度 → 更平滑的注意力分布
        #   想象一个 patch 与多个原型的相似度分数为 [0.5, 0.4, 0.3, 0.2]：

        #   - 低温 (τ=0.1)：softmax 后约 [0.66, 0.24, 0.08, 0.02] → 只关注最相似的原型
        #   - 高温 (τ=10)：softmax 后约 [0.27, 0.26, 0.25, 0.22] → 平等对待所有原型

        #   为什么设为可学习参数？

        #   模型可以在训练过程中自动调整：
        #   - 若需要选择性聚焦特定原型 → 降低温度
        #   - 若需要综合利用多个原型 → 提高温度

        self.temp_struct = nn.Parameter(torch.tensor(1.0))  # 文本分支温度
        self.temp_vis = nn.Parameter(torch.tensor(1.0))     # 视觉分支温度

        # ========== 交叉注意力模块  ==========
        self.num_heads = num_heads
        self.feature_dim = dim
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        # 🌟 创新点二（进阶版）：视觉引导的门控机制 (Gated Mechanism)
        # 1. 门控权重生成器 (Gate): 输出 0~1 之间的值，决定放行多少信息
        self.query_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()  
        )
        # 2. 候选信息提取器 (Update): 提取视觉和文本碰撞后的新特征
        self.query_update = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh()     # 使用 Tanh 将特征约束在 -1 到 1 之间，增强非线性
        )
        # 3. 归一化层: 防止融合后的特征方差过大，稳定训练
        self.query_norm = nn.LayerNorm(dim)

        # ========== 分类头 ==========
        # 将聚合后的bag特征映射到类别logits
        self.classification_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, 1)  # 二分类输出1个logit
        )

    def cross_attention(self, queries, keys, values, attention_mask=None):
        """
        多头交叉注意力

        参数:
            queries: (B, Q, D) 查询向量，这里是bag级文本原型
            keys: (B, K, D) 键向量，这里是融合后的patch特征
            values: (B, K, D) 值向量，与键相同

        返回:
            output: (B, Q, D) 注意力输出

        工作原理:
            Attention(Q, K, V) = softmax(QK^T/√d) * V

            这里bag级文本原型作为Query，"询问"patch特征中最相关的信息
        """
        bsz, q_len, _ = queries.size()
        _, kv_len, _ = keys.size()

        # 线性投影
        query_states = self.q_proj(queries)
        key_states = self.k_proj(keys)
        value_states = self.v_proj(values)

        # 重塑为多头格式: (B, num_heads, seq_len, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算缩放点积注意力
        # PyTorch内置的优化实现，支持flash attention加速
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask
        )

        # 重塑回原始格式: (B, Q, D)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.feature_dim)

        # 输出投影
        attn_output = self.o_proj(attn_output)

        return attn_output

    def compute_loss(self, logits, labels):
        """
        计算分类损失

        参数:
            logits: (B, num_classes) 模型输出
            labels: (B,) 真实标签

        返回:
            loss: 交叉熵损失
        """
        loss_cls = F.cross_entropy(logits, labels)

        return loss_cls

    def forward(self, V_patch, labels=None):
        """
        前向传播

        参数:
            V_patch: (B, N, D) 输入特征
                     B: batch size (通常为1，因为不同WSI的patch数不同)
                     N: patch数量 (每个WSI可能有几千个patch)
                     D: 特征维度 (512)
            labels: (B,) 标签 (可选，训练时提供)

        返回:
            dict: {
                'logits': (B, num_classes) 分类logits,
                'loss': 损失值 (如果提供labels)
            }

        处理流程:
            1. 特征投影: 将输入特征和原型投影到统一空间
            2. 双分支相似度计算: 分别计算视觉和文本相似度
            3. 最优传输融合: 对齐两个相似度分布
            4. 特征聚合: 使用交叉注意力聚合bag级别特征
            5. 分类: 输出类别预测
        """
        B, N, D = V_patch.shape

        # ========== 1. 特征投影 ==========
        # 将patch特征投影到原型空间
        V_proj = self.proj_v(V_patch)
        # 将视觉原型投影到相同空间
        P_vis = self.proj_v(self.proto_vis)

        # 将文本原型投影到原型空间
        prompt_struct = self.proj_llm(self.prompt_inst)
        prompt_bag = self.proj_llm(self.prompt_bag)

        # ========== 2. 计算实例到原型的相似度 ==========
        # 文本分支: 计算每个patch与每个文本原型的相似度（寻找符合病理描述的区域）
        # attn_struct[b,n,k] = 第b个样本第n个patch对第k个文本原型的注意力
        attn_struct = torch.einsum("bnd,kd->bnk", V_proj, prompt_struct) / self.temp_struct.exp()
        attn_struct = F.softmax(attn_struct, dim=-1)  # 归一化

        # 视觉分支: 计算每个patch与每个视觉原型的相似度
        # attn_vis[b,n,k] = 第b个样本第n个patch对第k个视觉原型的注意力
        attn_vis = torch.einsum("bnd,kd->bnk", V_proj, P_vis) / self.temp_vis.exp()
        attn_vis = F.softmax(attn_vis, dim=-1)  # 归一化

        # ========== 3. 最优传输融合 (SOT核心) ==========
        # 计算视觉原型和文本原型之间的代价矩阵
        # cost[i,j] = 视觉原型i和文本原型j之间的余弦距离
        cost_matrix = pairwise_cosine_distance(prompt_struct, P_vis)

        # 使用Sinkhorn算法求解最优传输
        # T[b,n,i,j] = 传输方案，决定如何对齐两个分支
        T = sinkhorn_ot(attn_struct, attn_vis, cost_matrix, epsilon=self.ot_epsilon, n_iters=self.ot_iter)
        # 通过传输矩阵融合注意力分布
        # 将实例到文本原型的注意力"传输"到视觉原型空间
        attn_fused = torch.einsum("bnij->bnj", T)
        # attn_fused = F.softmax(attn_fused, dim=-1)
        attn_fused = F.normalize(attn_fused, p=1, dim=-1) # L1归一化保证权重和为1

        # 使用融合后的注意力加权patch特征
        # patch_fused[b,n,:] = Σ_k attn_fused[b,n,k] * V_proj[b,n,:]
        patch_fused = torch.einsum("bnk,bnd->bnd", attn_fused, V_proj)

        # # ========== 4. 交叉注意力聚合 ==========
        # # bag级文本原型作为Query，查询最相关的patch信息
        # prompt_bag = prompt_bag.unsqueeze(0)  # (1, C, D) 添加batch维度
        # # 用全局文本去向融合后的图像 Patch 提问
        # cross_attn_output = self.cross_attention(
        #     queries=prompt_bag,
        #     keys=patch_fused,
        #     values=patch_fused
        # )
        
        
        # ========== 4. 视觉条件化的门控动态交叉注意力 (Gated Dynamic Query) ==========
        
        # 静态文本 Query
        prompt_bag_expanded = prompt_bag.unsqueeze(0).expand(B, -1, -1)
        
        # 4.1 提取当前 WSI 的全局视觉上下文 (平均池化)
        global_vis_context = patch_fused.mean(dim=1)
        
        # 4.2 扩展视觉特征以匹配文本 Query 的形状
        global_vis_expanded = global_vis_context.unsqueeze(1).expand(-1, self.C, -1)
        
        # 4.3 拼接特征 (用于计算门控和更新值)
        concat_query = torch.cat([prompt_bag_expanded, global_vis_expanded], dim=-1)
        
        # 4.4 门控机制核心：计算门控阀门 (Gate) 和 候选更新信息 (Update)
        gate = self.query_gate(concat_query)           # 形状: (B, C, D), 值域 [0, 1]
        update_info = self.query_update(concat_query)  # 形状: (B, C, D), 值域 [-1, 1]
        
        # 4.5 门控残差融合 (极其关键)：
        # 公式: Dynamic_Query = 原文本特征 + 阀门 * 候选新特征
        # 这样即使视觉特征是纯噪声，模型也可以学到让 gate=0，从而安全退化为原版模型
        dynamic_prompt_bag = prompt_bag_expanded + gate * update_info
        dynamic_prompt_bag = self.query_norm(dynamic_prompt_bag) 
        
        # 4.6 用门控进化后的“动态 Query”去向视觉特征收网
        cross_attn_output = self.cross_attention(
            queries=dynamic_prompt_bag, # (B, C, D) 带有宏观底色的动态搜查令
            keys=patch_fused,# (B, N, D) 包含上万个原始 patch 差异性的线索
            values=patch_fused# (B, N, D)
        )
        
        # ========== 5. 分类与损失计算 ==========
        # (保持原版分类逻辑，不使用 mean，防止 squeeze 报错)
        logits = self.classification_head(cross_attn_output).squeeze(-1)
        loss_cls = F.cross_entropy(logits, labels)

        return {
            'logits': logits,
            'loss': loss_cls,
            'loss_cls': loss_cls.detach()
        }
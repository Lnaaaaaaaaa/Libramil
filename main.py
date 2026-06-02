"""
Libra-MIL: 多模态原型立体融合模型训练脚本
用于少样本学习任务的全切片图像(WSI)分类
"""

# ==================== 导入依赖库 ====================
import os
import torch
import numpy as np
from tqdm import tqdm  # 进度条显示
from transformers import get_scheduler  # 学习率调度器
import json
import pandas as pd

# 项目内部模块
from models.mil_ot import MIL_MultiPrompt_OTFusion  # MIL模型与最优传输融合
from models.text_encoder import TextEncoder  # 文本编码器
from utils.general import compute_metrics, CSVWriter, write_summary_log  # 评估工具
from datasets.dataset import get_dataloader  # 数据加载器

# ==================== 命令行参数配置 ====================
def get_config():
    """
    解析命令行参数，配置训练超参数和路径设置

    Returns:
        args: 包含所有配置参数的命名空间对象
    """
    import argparse
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')

    # ---------- 数据路径配置 ----------
    parser.add_argument('--data_split_json', type=str, default=None, help='数据集划分JSON文件路径')
    parser.add_argument('--data_csv', type=str, default=None, help='数据集CSV文件路径')
    parser.add_argument('--h5_file_dir', type=str, default=None, help='WSI patch特征的h5文件目录')
    parser.add_argument('--instance_path', type=str, default=None, help='实例级结构化提示文本文件路径')
    parser.add_argument('--bag_path', type=str, default=None, help='包级结构化提示文本文件路径')
    parser.add_argument('--save_dir', type=str, default=None, help='模型和日志保存目录')

    # ---------- 模型超参数 ----------
    parser.add_argument('--feats_dim', type=int, default=512, help='特征维度')
    parser.add_argument('--num_struct_prompts', type=int, default=4, help='结构化提示数量')
    parser.add_argument('--num_vis_prototypes', type=int, default=4, help='视觉原型数量')
    parser.add_argument('--num_classes', type=int, default=2, help='分类类别数')
    parser.add_argument('--pooling_type', type=str, default='attention',
                        choices=['gated_attention', 'attention', 'mean'],
                        help='MIL池化类型')
    parser.add_argument('--use_proj', type=bool, default=True, help='是否使用投影层')
    parser.add_argument('--ot_epsilon', type=float, default=0.05, help='最优传输的epsilon参数')
    parser.add_argument('--ot_iter', type=int, default=20, help='最优传输迭代次数')
    parser.add_argument('--text_model_weights_path', type=str, default=None, help='文本模型权重路径')

    # ---------- 训练超参数 ----------
    parser.add_argument('--flods', type=int, default=5, help='交叉验证折数')
    parser.add_argument('--epoches', type=int, default=20, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--seed', type=int, default=7, help='随机种子，确保可复现性')
    parser.add_argument('--patience', type=int, default=15, help='早停耐心值')

    args = parser.parse_args()
    return args


# ==================== 随机种子设置 ====================
def seed_torch(seed=7, device=None):
    """
    设置所有随机种子，确保实验可复现

    Args:
        seed: 随机种子值
        device: 计算设备(CPU/GPU)
    """
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # GPU随机种子设置
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 禁用cudnn的随机性和benchmark模式，确保确定性
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ==================== 训练函数 ====================
def train(model, dataloader, device, epoch, optimizer, lr_scheduler, num_classes):
    """
    单轮训练过程

    Args:
        model: MIL模型
        dataloader: 训练数据加载器
        device: 计算设备
        epoch: 当前轮次
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        num_classes: 类别数

    Returns:
        tuple: (准确率, AUC, F1分数)
    """
    model.train()  # 设置为训练模式
    y_true, y_pred, y_score = [], [], []

    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for feats, labels in loop:
        # 数据移动到设备
        feats = feats.to(device)
        labels = labels.to(device)

        # 前向传播
        res_dic = model(feats, labels)
        loss = res_dic['loss']
        logits = res_dic['logits']

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # 收集预测结果
        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())
        y_score.extend(logits.detach().cpu().numpy())

    y_score = np.array(y_score)
    return compute_metrics(y_true, y_pred, y_score, num_classes)

# ==================== 评估函数 ====================
def evaluate(model, dataloader, device, epoch, num_classes):
    """
    模型评估过程（验证/测试）

    Args:
        model: MIL模型
        dataloader: 数据加载器
        device: 计算设备
        epoch: 当前轮次
        num_classes: 类别数

    Returns:
        tuple: (准确率, AUC, F1分数)
    """
    model.eval()  # 设置为评估模式
    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():  # 禁用梯度计算
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for feats, labels in loop:
            # 数据移动到设备
            feats = feats.to(device)
            labels = labels.to(device)

            # 前向传播
            res_dic = model(feats, labels)
            logits = res_dic['logits'].detach()

            # 计算预测概率和类别
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    y_score = np.array(y_score)
    return compute_metrics(y_true, y_pred, y_score, num_classes)

# ==================== 主函数 ====================
def main():
    """
    主训练流程：
    1. 加载配置和初始化环境
    2. 编码文本提示（结构化提示和包级提示）
    3. K折交叉验证训练
    4. 保存模型和日志
    """
    # ---------- 初始化配置 ----------
    args = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_torch(args.seed, device)

    # ---------- 加载文本提示 ----------
    # 加载实例级结构化提示（JSON格式）
    with open(args.instance_path, "r") as f:
        struct_llm = json.load(f)
    # 加载包级提示（CSV格式，跳过前num_classes行）
    bag_llm = pd.read_csv(args.bag_path, header=None)[0].tolist()[args.num_classes:]

    # 根据实际加载的提示数量更新配置
    args.num_struct_prompts = len(struct_llm)

    # ---------- 编码文本提示 ----------
    text_encoder = TextEncoder(args.feats_dim, args.text_model_weights_path)
    T_struct_llm = text_encoder(struct_llm).to(device)  # 结构化提示编码
    T_bag_llm = text_encoder(bag_llm).to(device)        # 包级提示编码

    # ---------- 创建输出目录 ----------
    os.makedirs(os.path.join(args.save_dir ,'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir ,'logs'), exist_ok=True)

    # 创建最终结果日志写入器
    final_csv = CSVWriter(filename=os.path.join(args.save_dir ,'logs', 'final_log.csv'), header=[
        'flod', 'test_acc', 'test_auc', 'test_f1_score'
        ])

    # ---------- K折交叉验证训练 ----------
    for fi in range(args.flods):
        # 每个fold重新设置种子，确保shuffle顺序独立且可复现
        seed_torch(args.seed + fi, device)

        # 初始化模型
        model = MIL_MultiPrompt_OTFusion(
            args.feats_dim,
            args.num_struct_prompts,
            args.num_vis_prototypes,
            args.num_classes,
            T_struct_llm,
            T_bag_llm,
            args.pooling_type,
            args.use_proj,
            args.ot_epsilon,
            args.ot_iter
        ).to(device)

        # 获取数据加载器
        loaders = get_dataloader(args.data_split_json, args.data_csv, args.h5_file_dir, idx=fi)

        # 配置优化器和学习率调度器
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        num_training_steps = len(loaders['train']) * args.epoches
        lr_scheduler = get_scheduler("cosine", optimizer=optimizer,
                                     num_warmup_steps=100,
                                     num_training_steps=num_training_steps)

        # 训练状态变量
        best_acc = 0.0
        best_auc = 0.0
        final_epoch = 1
        no_improve_count = 0  # 早停计数器

        # 创建每折训练日志
        header=[
            'epoch', 'final_epoch', 'train_acc', 'train_auc', 'train_f1_score',
            'valid_acc', 'valid_auc', 'valid_f1_score',
            'test_acc', 'test_auc', 'test_f1_score'
        ]
        log_csv = CSVWriter(filename=os.path.join(args.save_dir ,'logs', f'log_{fi}.csv'), header=header)

        # ---------- 单折训练循环 ----------
        for epoch in range(args.epoches):
            # 训练、验证、测试
            train_acc, train_auc, train_f1_score = train(model, loaders['train'], device, epoch, optimizer, lr_scheduler, args.num_classes)
            valid_acc, valid_auc, valid_f1_score = evaluate(model, loaders['valid'], device, epoch, args.num_classes)
            test_acc, test_auc, test_f1_score = evaluate(model, loaders['test'], device, epoch, args.num_classes)

            # 模型保存与早停判断
            if valid_acc > best_acc:
                # 验证集准确率提升，保存模型
                best_acc = valid_acc
                final_epoch = epoch + 1
                no_improve_count = 0
                torch.save(model.state_dict(), os.path.join(args.save_dir ,f'checkpoints/best_model_{fi}.pt'))
            elif valid_acc == best_acc and valid_auc > best_auc:
                # 准确率相同但AUC提升，保存模型
                best_auc = valid_auc
                final_epoch = epoch + 1
                no_improve_count = 0
                torch.save(model.state_dict(), os.path.join(args.save_dir ,f'checkpoints/best_model_{fi}.pt'))
            else:
                # 无改善，增加早停计数
                no_improve_count += 1
                if no_improve_count >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
                    break

            # 记录本轮结果
            log_csv.write_row([epoch+1, final_epoch, train_acc, train_auc, train_f1_score,
                            valid_acc, valid_auc, valid_f1_score,
                            test_acc, test_auc, test_f1_score
                            ])

        # ---------- 加载最佳模型并评估 ----------
        model.load_state_dict(torch.load(os.path.join(args.save_dir ,f'checkpoints/best_model_{fi}.pt')))
        test_acc, test_auc, test_f1_score = evaluate(model, loaders['test'], device, final_epoch, args.num_classes)
        final_csv.write_row([
            fi, test_acc, test_auc, test_f1_score
        ])

    # ---------- 输出汇总结果 ----------
    write_summary_log(
        os.path.join(args.save_dir ,'logs', 'final_log.csv'),
        os.path.join(args.save_dir ,'logs', 'summary_log.csv')
    )

# ==================== 程序入口 ====================
if __name__ == "__main__":
    main()

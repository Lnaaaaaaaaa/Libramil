"""
Libra-MIL 数据集模块 (增强版)
==============================
负责加载和处理全切片图像(WSI)的特征数据

数据流程:
    1. 预处理阶段：WSI图像 → 切片(patch) → 预训练模型提取特征 → 保存为h5文件
    2. 训练阶段：从h5文件加载特征 → 构建DataLoader

MIL (Multiple Instance Learning) 数据结构:
    - Bag（袋子）：一个WSI样本，包含多个patch特征
    - Instance（实例）：单个patch的特征向量
    - Label：只有Bag级别的标签，没有Instance级别的标签

例如：一个WSI可能有几千个patch，但只有一个"良性/恶性"的标签

支持的目录结构:
    1. 平铺结构: h5_file_dir/*.h5
    2. 分子目录结构: h5_file_dir/TCGA-KICH/*.h5, h5_file_dir/TCGA-KIRC/*.h5, ...

更新日期: 2026-05-25
"""

import os
import h5py
import pandas as pd
import random
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import json
from tqdm import tqdm


def build_h5_path_map(h5_file_dir):
    """
    构建样本名到h5文件路径的映射

    功能:
        递归搜索目录下的所有h5文件，建立 {样本名: 文件路径} 的映射表
        支持平铺结构和分子目录结构

    参数:
        h5_file_dir: h5文件的根目录

    返回:
        name_to_path: {样本名(stem): 完整文件路径} 的字典

    示例:
        输入目录结构:
            features/
            ├── TCGA-KICH/
            │   ├── sample1.h5
            │   └── sample2.h5
            └── TCGA-KIRC/
                ├── sample3.h5
                └── sample4.h5

        返回:
            {
                'sample1': '/path/to/features/TCGA-KICH/sample1.h5',
                'sample2': '/path/to/features/TCGA-KICH/sample2.h5',
                'sample3': '/path/to/features/TCGA-KIRC/sample3.h5',
                'sample4': '/path/to/features/TCGA-KIRC/sample4.h5',
            }
    """
    h5_file_dir = os.path.abspath(h5_file_dir)
    name_to_path = {}

    # 递归搜索所有h5文件
    # rglob 会递归遍历所有子目录
    for root, dirs, files in os.walk(h5_file_dir):
        for file in files:
            if file.endswith('.h5'):
                # 使用文件名(不含扩展名)作为key
                sample_name = os.path.splitext(file)[0]
                full_path = os.path.join(root, file)
                name_to_path[sample_name] = full_path

    return name_to_path


class WSIDataset(Dataset):
    """
    全切片图像(WSI)数据集类 (增强版)

    继承自PyTorch的Dataset类，用于加载预提取的WSI特征

    数据格式:
        - h5文件: 每个WSI对应一个h5文件，包含其所有patch的特征
        - CSV文件: 包含样本名称和标签的映射

    特征维度:
        - features: (N, D) 其中N是该WSI的patch数量，D是特征维度(如512)

    增强功能:
        - 自动递归搜索子目录
        - 支持分子目录结构
        - 支持增量添加新h5文件
    """

    def __init__(self, indices, label_csv, h5_file_dir):
        """
        初始化数据集

        参数:
            indices: 样本名称列表，指定要加载哪些WSI
            label_csv: 标签文件路径，包含 'name' 和 'label' 两列
            h5_file_dir: h5特征文件的存储目录 (支持分子目录结构)
        """
        self.h5_file_dir = h5_file_dir

        # 读取标签文件，构建 {样本名: 标签} 的字典
        df = pd.read_csv(label_csv)
        self.name_label = df.set_index('name')['label'].to_dict()

        # ====================================================================
        # 关键改动: 构建样本名到文件路径的映射
        # 支持分子目录结构，无需手动创建平铺目录
        # ====================================================================
        print(f"  [信息] 扫描h5文件目录: {h5_file_dir}")
        self.name_to_path = build_h5_path_map(h5_file_dir)
        print(f"  [信息] 发现 {len(self.name_to_path)} 个h5文件")

        # 存储特征数据：{样本名: 特征张量}
        self.features = {}
        # 有效样本名称列表
        self.indices = []
        # 统计
        missing_files = []
        missing_labels = []

        # 遍历所有样本，加载特征
        for name in tqdm(indices, desc="Loading WSI features"):
            # 检查文件是否存在
            if name not in self.name_to_path:
                missing_files.append(name)
                continue

            # 检查标签是否存在
            if name not in self.name_label:
                missing_labels.append(name)
                continue

            try:
                # 获取h5文件的完整路径
                h5_path = self.name_to_path[name]

                # 从h5文件加载特征
                # h5文件结构: {'features': (N, D) 数组}
                with h5py.File(h5_path, 'r') as h5:
                    self.features[name] = torch.tensor(h5['features'][:], dtype=torch.float32)
                self.indices.append(name)
            except Exception as e:
                print(f'  [错误] 加载 {name} 失败: {e}')
                continue

        # 打印统计信息
        if missing_files:
            print(f"  [警告] {len(missing_files)} 个样本找不到h5文件")
        if missing_labels:
            print(f"  [警告] {len(missing_labels)} 个样本找不到标签")
        print(f"  [完成] 成功加载 {len(self.indices)} 个样本")

    def __len__(self):
        """返回数据集中的样本数量"""
        return len(self.indices)

    def __getitem__(self, idx):
        """
        获取单个样本

        参数:
            idx: 样本索引

        返回:
            features: (N, D) 该WSI的所有patch特征
            label: 该WSI的标签（整数）

        注意:
            - 不同WSI的patch数量N可能不同
            - 这正是MIL的特点：处理变长实例数量
        """
        name = self.indices[idx]
        label = self.name_label[name]
        features = self.features[name]

        return features, label


def get_dataloader(data_split_json, data_csv, h5_file_dir, idx=0):
    """
    构建数据加载器

    参数:
        data_split_json: 数据划分文件路径，包含训练/验证/测试集的样本名
        data_csv: 标签文件路径
        h5_file_dir: 特征文件目录 (支持分子目录结构，会递归搜索子目录)
        idx: 折索引，用于k折交叉验证（默认为第0折）

    返回:
        字典: {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

    数据划分说明:
        - JSON文件格式: {'train_0': [...], 'val_0': [...], 'test_0': [...], ...}
        - 支持k折交叉验证，每折独立划分

    DataLoader参数说明:
        - batch_size=1: 每个batch包含一个WSI（因为不同WSI的patch数不同）
        - shuffle=True: 训练集打乱顺序
        - num_workers: 数据加载的并行进程数

    使用示例:
        # 分子目录结构
        loaders = get_dataloader(
            data_split_json='./data/data_split.json',
            data_csv='./data/labels.csv',
            h5_file_dir='/path/to/features',  # 包含 TCGA-KICH/, TCGA-KIRC/ 等子目录
            idx=0
        )
    """
    # 读取数据划分配置
    with open(data_split_json, 'r') as fp:
        indices = json.load(fp)

        # 为训练、验证、测试集创建Dataset
        print(f"\n[数据加载] Fold {idx}")
        print(f"  训练集: {len(indices[f'train_{idx}'])} 样本")
        print(f"  验证集: {len(indices[f'val_{idx}'])} 样本")
        print(f"  测试集: {len(indices[f'test_{idx}'])} 样本")

        train_set = WSIDataset(indices[f'train_{idx}'], data_csv, h5_file_dir)
        valid_set = WSIDataset(indices[f'val_{idx}'], data_csv, h5_file_dir)
        test_set  = WSIDataset(indices[f'test_{idx}'], data_csv, h5_file_dir)

    # 创建DataLoader
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=1)
    test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False, num_workers=1)

    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

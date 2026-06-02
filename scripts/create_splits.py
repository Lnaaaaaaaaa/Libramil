#!/usr/bin/env python3
"""
Libra-MIL 数据划分脚本 (增强版)
================================

功能:
    1. 支持多种目录结构:
       - 分子目录结构: features/TCGA-KICH/*.h5, features/TCGA-KIRC/*.h5, ...
       - 平铺结构: features/*.h5
    2. 增量更新: 支持在现有划分基础上添加新样本
    3. 分层抽样: 确保每折中各类别比例一致
    4. 少样本模式: 通过 --shot 参数控制

使用方法:
    # 全量模式 (默认，首次创建)
    python create_splits.py --features_dir /path/to/features --output_dir ./data

    # 增量更新 (添加新样本时)
    python create_splits.py --features_dir /path/to/features --output_dir ./data --incremental

    # 强制重新划分
    python create_splits.py --features_dir /path/to/features --output_dir ./data --force

    # 少样本模式 (如 10-shot)
    python create_splits.py --features_dir /path/to/features --output_dir ./data --shot 10

作者: Libra-MIL 项目
更新日期: 2026-05-25
"""

import os
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from sklearn.model_selection import StratifiedKFold


# ============================================================================
# 标签映射配置
# ============================================================================

# TCGA-RCC 三分类标签映射
# 说明: 根据数据集子目录名称确定样本标签
LABEL_MAP = {
    'TCGA-KICH': 0,  # 肾嫌色细胞癌 (chromophobe RCC)
    'TCGA-KIRC': 1,  # 肾透明细胞癌 (clear cell RCC)
    'TCGA-KIRP': 2,  # 肾乳头状细胞癌 (papillary RCC)
}

# 文件名前缀到标签的映射 (用于平铺目录结构)
# 说明: 当所有h5文件在同一目录时，通过文件名前缀判断类别
FILE_PREFIX_MAP = {
    # KICH (label=0)
    'TCGA-KL': 0,
    # KIRC (label=1) - 包含多种前缀
    'TCGA-KM': 1, 'KM-': 1,
    'TCGA-KO': 1, 'KO-': 1,
    'TCGA-KN': 1, 'KN-': 1,
    'TCGA-UW': 1, 'UW-': 1,
    'TCGA-B': 1,
    'TCGA-A': 1,
    'TCGA-C': 1,
    'TCGA-6D': 1,
    # KIRP (label=2) - 如有需要可添加
}


def get_label_from_dataset(dataset_name):
    """
    根据数据集名称获取标签

    参数:
        dataset_name: 数据集名称，如 'TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP'

    返回:
        标签值 (0, 1, 2)，如果未知则返回 -1
    """
    return LABEL_MAP.get(dataset_name, -1)


def get_label_from_filename(filename):
    """
    根据文件名前缀获取标签 (用于平铺目录结构)

    参数:
        filename: h5文件名 (不含扩展名)

    返回:
        (label, dataset_name) 元组，如果无法判断则返回 (-1, None)
    """
    # 首先尝试通过前缀匹配
    for prefix, label in FILE_PREFIX_MAP.items():
        if filename.startswith(prefix) or (prefix.endswith('-') and prefix in filename):
            dataset_name = [k for k, v in LABEL_MAP.items() if v == label][0] if label in LABEL_MAP.values() else 'Unknown'
            return label, dataset_name

    # 其次尝试通过文件名中的关键字匹配
    if 'KICH' in filename:
        return 0, 'TCGA-KICH'
    elif 'KIRC' in filename:
        return 1, 'TCGA-KIRC'
    elif 'KIRP' in filename:
        return 2, 'TCGA-KIRP'

    return -1, None


def collect_samples(features_dir):
    """
    收集所有样本及其标签

    支持两种目录结构:
        1. 分子目录结构: features/TCGA-KICH/*.h5, features/TCGA-KIRC/*.h5, ...
        2. 平铺结构: features/*.h5 (通过文件名前缀判断类别)

    参数:
        features_dir: 特征文件根目录路径

    返回:
        samples: [{'name': 样本名, 'label': 标签, 'dataset': 数据集名, 'path': 文件路径}, ...]
    """
    features_dir = Path(features_dir)
    samples = []

    # 检查是否有子目录结构 (TCGA-* 子目录)
    subdirs = [d for d in features_dir.iterdir() if d.is_dir() and d.name.startswith('TCGA')]

    if subdirs:
        # ====================================================================
        # 结构1: 分子目录结构
        # ====================================================================
        print(f"[信息] 检测到分子目录结构: {[d.name for d in subdirs]}")

        for subdir in subdirs:
            dataset_name = subdir.name
            label = get_label_from_dataset(dataset_name)

            if label == -1:
                print(f"  [警告] 未知数据集 {dataset_name}，跳过")
                continue

            # 收集该子目录下的所有h5文件
            h5_files = list(subdir.glob('*.h5'))

            for h5_file in h5_files:
                samples.append({
                    'name': h5_file.stem,      # 文件名 (不含扩展名)
                    'label': label,            # 类别标签
                    'dataset': dataset_name,   # 数据集名称
                    'path': str(h5_file)       # 完整路径
                })

            print(f"  [收集] {dataset_name}: {len(h5_files)} 个样本 (label={label})")

    else:
        # ====================================================================
        # 结构2: 平铺目录结构
        # ====================================================================
        print("[信息] 检测到平铺目录结构")

        h5_files = list(features_dir.glob('*.h5'))

        for h5_file in h5_files:
            name = h5_file.stem
            label, dataset = get_label_from_filename(name)

            if label == -1:
                print(f"  [警告] 无法判断类别: {name}，跳过")
                continue

            samples.append({
                'name': name,
                'label': label,
                'dataset': dataset,
                'path': str(h5_file)
            })

    return samples


def load_existing_split(output_dir):
    """
    加载已有的数据划分文件 (用于增量更新)

    参数:
        output_dir: 输出目录路径

    返回:
        existing_samples: 已有样本的字典 {样本名: 样本信息}
        existing_split: 已有的划分数据
    """
    output_dir = Path(output_dir)
    split_path = output_dir / 'data_split.json'
    labels_path = output_dir / 'labels.csv'

    existing_samples = {}
    existing_split = None

    # 加载已有划分
    if split_path.exists():
        with open(split_path, 'r') as f:
            existing_split = json.load(f)
        print(f"[增量] 加载已有划分: {split_path}")

    # 加载已有标签
    if labels_path.exists():
        with open(labels_path, 'r') as f:
            lines = f.readlines()[1:]  # 跳过表头
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    name, label = parts[0], int(parts[1])
                    existing_samples[name] = {'name': name, 'label': label}
        print(f"[增量] 已有样本数: {len(existing_samples)}")

    return existing_samples, existing_split


def create_stratified_splits(samples, n_folds=5, shot=None, seed=42):
    """
    创建分层5折交叉验证划分

    参数:
        samples: 样本列表 [{'name': ..., 'label': ..., ...}, ...]
        n_folds: 折数 (默认5折)
        shot: 少样本设置，None表示全量
        seed: 随机种子

    返回:
        data_split: {'train_0': [...], 'val_0': [...], 'test_0': [...], ...}

    少样本设置说明:
        - shot=None: 全量模式，使用所有样本进行5折交叉验证
        - shot=K: 少样本模式，每类选择K个样本作为训练集，剩余样本用于验证/测试
                  然后对剩余样本进行5折交叉验证
    """
    random.seed(seed)

    # 按类别分组
    class_samples = defaultdict(list)
    for s in samples:
        class_samples[s['label']].append(s['name'])

    # 打印类别分布
    print(f"\n[统计] 类别分布:")
    for label, names in sorted(class_samples.items()):
        # 获取该类别对应的数据集名称
        dataset_names = set()
        for s in samples:
            if s['name'] in names:
                dataset_names.add(s['dataset'])
        print(f"  Label {label}: {len(names)} 个样本 ({dataset_names})")

    # ========================================================================
    # 少样本模式: 训练集固定为每类K个样本，剩余样本用于验证/测试
    # ========================================================================
    if shot is not None:
        print(f"\n[少样本] 设置: {shot}-shot")

        train_samples = {}      # 每类的训练样本
        remaining_samples = {}  # 每类的剩余样本(用于验证/测试)

        for label, names in class_samples.items():
            if len(names) >= shot:
                # 随机选择K个样本作为训练集
                train_names = random.sample(names, shot)
                # 剩余样本用于验证/测试
                remaining_names = [n for n in names if n not in train_names]
            else:
                # 样本不足，使用全部作为训练集
                train_names = names
                remaining_names = []
                print(f"  [警告] Label {label} 样本数不足 {shot}，使用全部 {len(names)} 个样本")

            train_samples[label] = train_names
            remaining_samples[label] = remaining_names

            print(f"  Label {label}: 训练 {len(train_names)} 个, 剩余 {len(remaining_names)} 个")

        # 对剩余样本进行5折交叉验证划分(用于验证和测试)
        data_split = {}

        for fold in range(n_folds):
            fold_val_names = []
            fold_test_names = []

            for label, names in remaining_samples.items():
                if len(names) < 2:
                    # 剩余样本太少，全部作为测试集
                    fold_test_names.extend(names)
                    continue

                n = len(names)
                indices = list(range(n))
                random.shuffle(indices)

                # 验证集和测试集各占一半
                val_size = n // 2

                # 不同fold使用不同的划分
                val_start = (fold * val_size) % n
                val_end = val_start + val_size
                if val_end > n:
                    val_indices = indices[val_start:] + indices[:val_end - n]
                    test_indices = indices[val_end - n:val_start]
                else:
                    val_indices = indices[val_start:val_end]
                    test_indices = indices[:val_start] + indices[val_end:]

                fold_val_names.extend([names[i] for i in val_indices])
                fold_test_names.extend([names[i] for i in test_indices])

            # 训练集固定为每类K个样本
            all_train_names = []
            for label, names in train_samples.items():
                all_train_names.extend(names)

            random.shuffle(all_train_names)
            random.shuffle(fold_val_names)
            random.shuffle(fold_test_names)

            data_split[f'train_{fold}'] = all_train_names
            data_split[f'val_{fold}'] = fold_val_names
            data_split[f'test_{fold}'] = fold_test_names

            print(f"\n[划分] Fold {fold}:")
            print(f"  训练集: {len(all_train_names)}")
            print(f"  验证集: {len(fold_val_names)}")
            print(f"  测试集: {len(fold_test_names)}")

        return data_split

    # ========================================================================
    # 全量模式: 标准的5折交叉验证
    # ========================================================================
    # 为每个类别创建分层索引
    class_indices = {}
    for label, names in class_samples.items():
        n = len(names)
        indices = list(range(n))
        random.shuffle(indices)
        class_indices[label] = indices

    # 构建数据划分
    data_split = {}

    for fold in range(n_folds):
        train_names = []
        val_names = []
        test_names = []

        for label, names in class_samples.items():
            n = len(names)
            indices = class_indices[label]

            # 计算每折的测试集索引范围
            test_size = n // n_folds
            test_start = fold * test_size
            test_end = test_start + test_size if fold < n_folds - 1 else n

            test_indices = set(indices[test_start:test_end])

            # 验证集取测试集的下一折
            val_fold = (fold + 1) % n_folds
            val_start = val_fold * test_size
            val_end = val_start + test_size if val_fold < n_folds - 1 else n
            val_indices = set(indices[val_start:val_end])

            # 训练集为剩余样本
            train_indices = [i for i in indices if i not in test_indices and i not in val_indices]

            # 收集样本名
            train_names.extend([names[i] for i in train_indices])
            val_names.extend([names[i] for i in val_indices])
            test_names.extend([names[i] for i in test_indices])

        # 打乱顺序
        random.shuffle(train_names)
        random.shuffle(val_names)
        random.shuffle(test_names)

        # 保存划分
        data_split[f'train_{fold}'] = train_names
        data_split[f'val_{fold}'] = val_names
        data_split[f'test_{fold}'] = test_names

        print(f"\n[划分] Fold {fold}:")
        print(f"  训练集: {len(train_names)}")
        print(f"  验证集: {len(val_names)}")
        print(f"  测试集: {len(test_names)}")

    return data_split


def incremental_update(existing_split, new_samples, n_folds=5, seed=42):
    """
    增量更新数据划分

    功能:
        - 保留已有样本的划分关系
        - 将新样本按分层方式添加到训练集中
        - 如需重新划分，请使用 --force 参数

    参数:
        existing_split: 已有的划分数据
        new_samples: 新样本列表
        n_folds: 折数
        seed: 随机种子

    返回:
        data_split: 更新后的划分数据
    """
    random.seed(seed)

    # 获取已有样本集合
    existing_names = set()
    for fold in range(n_folds):
        existing_names.update(existing_split.get(f'train_{fold}', []))
        existing_names.update(existing_split.get(f'val_{fold}', []))
        existing_names.update(existing_split.get(f'test_{fold}', []))

    # 筛选真正的新样本
    truly_new = [s for s in new_samples if s['name'] not in existing_names]

    if not truly_new:
        print("[增量] 没有发现新样本")
        return existing_split

    print(f"[增量] 发现 {len(truly_new)} 个新样本")

    # 按类别分组新样本
    new_by_class = defaultdict(list)
    for s in truly_new:
        new_by_class[s['label']].append(s['name'])

    # 将新样本添加到各折的训练集中
    data_split = existing_split.copy()

    for fold in range(n_folds):
        train_key = f'train_{fold}'
        current_train = data_split.get(train_key, [])

        # 按比例添加新样本到训练集
        for label, names in new_by_class.items():
            # 简单策略: 将所有新样本添加到训练集
            # 这样可以避免破坏已有的验证/测试集分布
            current_train.extend(names)

        random.shuffle(current_train)
        data_split[train_key] = current_train

    return data_split


def create_labels_csv(samples, output_path, existing_samples=None):
    """
    创建或更新标签CSV文件

    参数:
        samples: 当前收集的所有样本
        output_path: 输出文件路径
        existing_samples: 已有的样本字典 (用于增量更新)
    """
    # 合并已有样本和新增样本
    all_samples = {}

    if existing_samples:
        all_samples.update(existing_samples)

    for s in samples:
        all_samples[s['name']] = {'name': s['name'], 'label': s['label']}

    # 写入CSV
    with open(output_path, 'w') as f:
        f.write("name,label\n")
        for name in sorted(all_samples.keys()):
            s = all_samples[name]
            f.write(f"{s['name']},{s['label']}\n")

    print(f"\n[保存] 标签文件: {output_path}")
    print(f"  总样本数: {len(all_samples)}")


def main():
    # ========================================================================
    # 参数解析
    # ========================================================================
    parser = argparse.ArgumentParser(
        description='Libra-MIL 数据划分脚本 (增强版)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 首次创建划分 (全量模式，默认)
  python create_splits.py --features_dir /path/to/features --output_dir ./data

  # 增量更新 (添加新样本)
  python create_splits.py --features_dir /path/to/features --output_dir ./data --incremental

  # 强制重新划分
  python create_splits.py --features_dir /path/to/features --output_dir ./data --force

  # 少样本模式
  python create_splits.py --features_dir /path/to/features --output_dir ./data --shot 10
        """
    )

    parser.add_argument('--features_dir', type=str, required=True,
                        help='特征文件目录路径 (支持分子目录或平铺结构)')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='输出目录 (默认: ./data)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='交叉验证折数 (默认: 5)')
    parser.add_argument('--shot', type=int, default=None,
                        help='少样本设置 (如 5, 10, 20)，None表示全量')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--incremental', action='store_true',
                        help='增量更新模式: 保留已有划分，只添加新样本到训练集')
    parser.add_argument('--force', action='store_true',
                        help='强制重新划分 (忽略已有划分)')

    args = parser.parse_args()

    # ========================================================================
    # 初始化
    # ========================================================================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Libra-MIL 数据划分脚本 (增强版)")
    print("=" * 70)
    print(f"[配置] 特征目录: {args.features_dir}")
    print(f"[配置] 输出目录: {args.output_dir}")
    print(f"[配置] 折数: {args.n_folds}")
    print(f"[配置] 少样本: {args.shot if args.shot else '全量'}")
    print(f"[配置] 随机种子: {args.seed}")
    print(f"[配置] 增量模式: {args.incremental}")
    print(f"[配置] 强制重划: {args.force}")
    print("=" * 70)

    # ========================================================================
    # 收集样本
    # ========================================================================
    print("\n[步骤1] 收集样本...")
    samples = collect_samples(args.features_dir)
    print(f"\n[结果] 总样本数: {len(samples)}")

    if len(samples) == 0:
        print("[错误] 未找到任何样本!")
        return

    # ========================================================================
    # 数据划分
    # ========================================================================
    print("\n[步骤2] 数据划分...")

    existing_samples = None
    existing_split = None

    # 增量更新模式: 加载已有划分
    if args.incremental and not args.force:
        existing_samples, existing_split = load_existing_split(args.output_dir)

        if existing_split:
            # 执行增量更新
            data_split = incremental_update(existing_split, samples, args.n_folds, args.seed)
        else:
            print("[增量] 未找到已有划分，执行全新划分")
            data_split = create_stratified_splits(samples, args.n_folds, args.shot, args.seed)
    else:
        # 全新划分
        if args.incremental:
            print("[信息] 强制重新划分模式，忽略 --incremental 参数")
        data_split = create_stratified_splits(samples, args.n_folds, args.shot, args.seed)

    # ========================================================================
    # 保存结果
    # ========================================================================
    print("\n[步骤3] 保存结果...")

    # 保存数据划分
    split_path = output_dir / 'data_split.json'
    with open(split_path, 'w') as f:
        json.dump(data_split, f, indent=2)
    print(f"[保存] 划分文件: {split_path}")

    # 保存标签文件
    labels_path = output_dir / 'labels.csv'
    create_labels_csv(samples, labels_path, existing_samples if args.incremental else None)

    # ========================================================================
    # 完成
    # ========================================================================
    print("\n" + "=" * 70)
    print("[完成] 数据划分成功!")
    print("=" * 70)
    print(f"输出文件:")
    print(f"  - {split_path}")
    print(f"  - {labels_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()

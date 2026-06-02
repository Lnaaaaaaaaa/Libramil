
"""
Libra-MIL WSI特征提取脚本 (CLAM版本 - 已修正组织分割与坐标Bug)
基于论文: Libra-MIL: Multimodal Prototypes Stereoscopic Infused with Task-specific Language Priors

使用CLAM框架进行WSI切片:
- Patch大小: 512x512像素 (论文指定)
- 放大倍数: 20x (论文指定)
- 组织分割: CLAM默认自适应参数 (已修正)
- 特征提取: CONCH视觉-语言基础模型

数据集:
- TCGA-KICH: 肾嫌色细胞癌
- TCGA-KIRC: 肾透明细胞癌
- TCGA-KIRP: 肾乳头状细胞癌
"""

import os
import sys
import argparse
import time
import json
import warnings
from pathlib import Path
from functools import partial
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
import torch
import h5py
from tqdm import tqdm
from queue import Queue

# 设置离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
# 解决显存碎片化问题
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# CLAM路径
CLAM_PATH = '/mnt/sda1/ln_workspace/CLAM'
sys.path.insert(0, CLAM_PATH)

# CONCH路径
CONCH_PATH = '/mnt/sda1/ln_workspace/CONCH'
sys.path.insert(0, CONCH_PATH)

# 导入CLAM
try:
    from wsi_core.WholeSlideImage import WholeSlideImage
    from wsi_core.batch_process_utils import initialize_df
except ImportError as e:
    print(f"错误: 无法导入CLAM模块 - {e}")
    sys.exit(1)

# 导入CONCH
try:
    from conch.open_clip_custom import create_model_from_pretrained
except ImportError as e:
    print(f"错误: 无法导入CONCH模块 - {e}")
    sys.exit(1)


class ProgressTracker:
    """进度追踪器"""

    def __init__(self, progress_file):
        self.progress_file = Path(progress_file)
        self.data = self._load()

    def _load(self):
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            'start_time': None,
            'current_dataset': None,
            'current_slide': None,
            'datasets': {},
            'total_slides': 0,
            'processed_slides': 0,
            'status': 'idle'
        }

    def save(self):
        self.data['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.progress_file, 'w') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def start(self, total_slides):
        self.data['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.data['total_slides'] = total_slides
        self.data['status'] = 'running'
        self.save()

    def update_dataset(self, dataset_name, total, processed):
        self.data['current_dataset'] = dataset_name
        if dataset_name not in self.data['datasets']:
            self.data['datasets'][dataset_name] = {
                'total': total,
                'processed': 0,
                'slides': {}
            }
        self.data['datasets'][dataset_name]['processed'] = processed
        self.save()

    def update_slide(self, dataset_name, slide_id, status, n_patches=None, time_elapsed=None):
        self.data['current_slide'] = slide_id
        self.data['processed_slides'] += 1
        if slide_id not in self.data['datasets'][dataset_name]['slides']:
            self.data['datasets'][dataset_name]['slides'][slide_id] = {}
        self.data['datasets'][dataset_name]['slides'][slide_id].update({
            'status': status,
            'n_patches': n_patches,
            'time': time_elapsed,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        self.save()

    def finish(self):
        self.data['status'] = 'completed'
        self.data['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.save()


class CLAMFeatureExtractor:
    """
    使用CLAM进行WSI切片 + CONCH特征提取
    
    流程:
    1. CLAM组织分割 (segmentTissue)
    2. CLAM切片 (process_contours)
    3. CONCH特征提取
    """

    def __init__(self, checkpoint_path, device='cuda', patch_size=512, patch_level=0):
        """
        Args:
            checkpoint_path: CONCH模型权重路径
            device: 计算设备
            patch_size: patch大小 (论文: 512x512)
            patch_level: WSI金字塔层级 (需要根据放大倍数确定)
        """
        self.device = device
        self.patch_size = patch_size
        self.patch_level = patch_level

        # CLAM参数 - 修正组织检测失败的关键配置
        self.seg_params = {
            'seg_level': -1,  # 自动选择
            'sthresh': 8,
            'mthresh': 7,
            'close': 4,
            'use_otsu': True,  # 开启大津自适应二值化
            'keep_ids': [],    # 必须是空列表，不是'none'字符串！
            'exclude_ids': []  # 必须是空列表，不是'none'字符串！
        }
        
        # 🟢 终极修正 1：放低过滤面积门槛，防止小轮廓或碎标本被全部过滤（解决Contours为0）
        self.filter_params = {
            'a_t': 1,          # 从 100 降低到 1，允许提取更小的组织区域
            'a_h': 1,          # 降低孔洞过滤面积
            'max_n_holes': 8
        }
        self.patch_params = {
            'use_padding': True,
            'contour_fn': 'four_pt'
        }

        print("=" * 60)
        print("初始化CLAM + CONCH特征提取器")
        print(f"  Patch大小: {patch_size}x{patch_size}")
        print(f"  设备: {device}")
        print("=" * 60)

        # 加载CONCH模型
        print("加载CONCH模型...")
        self.model, self.preprocess = create_model_from_pretrained(
            'conch_ViT-B-16',
            checkpoint_path
        )
        self.model = self.model.to(device)
        self.model.eval()
        self.model.forward = partial(self.model.encode_image, proj_contrast=False, normalize=False)
        print("CONCH模型加载完成!")

    def get_patch_level_for_20x(self, wsi):
        """
        获取对应20x放大倍数的patch_level
        
        [cite_start]20x 对应 mpp ≈ 0.5 μm/pixel [cite: 20]
        """
        import openslide
        
        # 获取WSI的mpp
        mpp = None
        try:
            mpp = float(wsi.properties.get(openslide.PROPERTY_NAME_MPP_X,
                                            wsi.properties.get(openslide.PROPERTY_NAME_MPP_Y, None)))
        except (TypeError, ValueError):
            pass

        if mpp is None:
            print("  警告: 无法获取MPP，使用默认level=0")
            return 0

        # [cite_start]目标mpp (20x ≈ 0.5) [cite: 20]
        target_mpp = 0.5
        best_level = 0
        min_diff = float('inf')

        for level in range(wsi.level_count):
            downsampling = wsi.level_downsamples[level]
            level_mpp = mpp * downsampling
            diff = abs(level_mpp - target_mpp)
            if diff < min_diff:
                min_diff = diff
                best_level = level

        actual_mpp = mpp * wsi.level_downsamples[best_level]
        actual_mag = 10 / actual_mpp
        print(f"  WSI mpp={mpp:.3f}, 选择level={best_level}, 实际≈{actual_mag:.1f}x")

        return best_level

    def segment_and_patch(self, slide_path, save_dir):
        """
        使用CLAM进行组织分割和切片
        """
        slide_path = Path(slide_path)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        slide_id = slide_path.stem
        h5_path = save_dir / f"{slide_id}.h5"

        if h5_path.exists():
            print(f"  切片文件已存在: {h5_path}")
            wsi_object = WholeSlideImage(str(slide_path))
            return str(h5_path), wsi_object

        print(f"  CLAM组织分割与切片...")
        wsi_object = WholeSlideImage(str(slide_path))

        # 🟢 终极修正 2：加固自动层级选择，防止残缺TCGA切片导致 tuple index out of range
        if self.seg_params['seg_level'] < 0:
            wsi = wsi_object.getOpenSlide()
            try:
                best_level = wsi.get_best_level_for_downsample(64)
                # 检查算出的层级是否越界
                if best_level >= wsi.level_count:
                    best_level = wsi.level_count - 1
            except:
                best_level = wsi.level_count - 1
            self.seg_params['seg_level'] = best_level

        # 组织分割
        wsi_object.segmentTissue(**self.seg_params, filter_params=self.filter_params)

        # 获取20x对应的patch_level
        self.patch_level = self.get_patch_level_for_20x(wsi_object.getOpenSlide())

        # 切片
        patch_params = self.patch_params.copy()
        patch_params.update({
            'patch_level': self.patch_level,
            'patch_size': self.patch_size,
            'step_size': self.patch_size,  # 非重叠
            'save_path': str(save_dir)
        })

        # CLAM的process_contours返回值有bug，不依赖它
        wsi_object.process_contours(**patch_params)

        # 直接检查文件是否创建成功
        expected_h5_path = save_dir / f"{slide_id}.h5"
        if expected_h5_path.exists():
            return str(expected_h5_path), wsi_object
        else:
            raise ValueError(f"CLAM未能从该切片中提取出任何有效组织区域！")

    def extract_features_from_h5(self, slide_path, h5_path, output_path, batch_size=64, num_workers=8):
        """
        从CLAM生成的h5坐标文件提取CONCH特征 (稳定优化版)

        优化策略:
        1. 多线程并行读取patch
        2. 定期清理显存碎片
        3. 错误处理更健壮
        """
        from PIL import Image
        import openslide

        slide_path = Path(slide_path)
        h5_path = Path(h5_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 读取坐标
        with h5py.File(h5_path, 'r') as f:
            coords = f['coords'][:]
            patch_level = f['coords'].attrs.get('patch_level', self.patch_level)
            patch_size = f['coords'].attrs.get('patch_size', self.patch_size)

        print(f"  坐标数: {len(coords)}, patch_size={patch_size}, level={patch_level}")

        if len(coords) == 0:
            print("  警告: 没有有效的patch坐标!")
            return False, 0

        # 打开WSI
        try:
            slide = openslide.open_slide(str(slide_path))
        except Exception as e:
            print(f"  错误: 无法打开WSI - {e}")
            return False, 0

        all_features = []
        valid_count = 0
        n_coords = len(coords)

        def load_single_patch(idx):
            """加载单个patch"""
            x, y = coords[idx]
            try:
                patch = slide.read_region(
                    (int(x), int(y)),
                    patch_level,
                    (patch_size, patch_size)
                ).convert('RGB')
                if patch_size != 512:
                    patch = patch.resize((512, 512), Image.BILINEAR)
                return self.preprocess(patch)
            except:
                return None

        # 使用线程池并行加载
        pbar = tqdm(total=n_coords, desc="  提取特征", leave=False)

        with torch.inference_mode():
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for batch_start in range(0, n_coords, batch_size):
                    batch_end = min(batch_start + batch_size, n_coords)
                    batch_indices = list(range(batch_start, batch_end))

                    # 并行加载
                    results = list(executor.map(load_single_patch, batch_indices))

                    # 过滤有效结果
                    valid_patches = [r for r in results if r is not None]
                    valid_count += len(valid_patches)
                    pbar.update(len(batch_indices))

                    if not valid_patches:
                        continue

                    # GPU推理
                    try:
                        batch_tensor = torch.stack(valid_patches).to(self.device)
                        features = self.model(batch_tensor)
                        all_features.append(features.cpu().numpy())
                        del batch_tensor, features
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print(f"\n  显存不足，尝试清理后继续...")
                            torch.cuda.empty_cache()
                            # 重试
                            batch_tensor = torch.stack(valid_patches).to(self.device)
                            features = self.model(batch_tensor)
                            all_features.append(features.cpu().numpy())
                            del batch_tensor, features
                        else:
                            raise e

        pbar.close()
        slide.close()

        # 清理显存
        torch.cuda.empty_cache()

        if not all_features:
            print("  警告: 没有提取到任何特征!")
            return False, 0

        # 合并特征
        all_features = np.vstack(all_features).astype(np.float32)

        # 保存
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('features', data=all_features, compression='gzip')
            f.create_dataset('coords', data=coords, compression='gzip')
            f['coords'].attrs['patch_level'] = patch_level
            f['coords'].attrs['patch_size'] = patch_size

        print(f"  特征形状: {all_features.shape}")
        print(f"  保存至: {output_path}")

        return True, valid_count

    def process_slide(self, slide_path, patches_dir, features_dir, batch_size=64):
        """
        处理单个slide: CLAM切片 + CONCH特征提取
        """
        slide_path = Path(slide_path)
        slide_id = slide_path.stem

        # [cite_start]CLAM切片 [cite: 588]
        h5_path, wsi_object = self.segment_and_patch(slide_path, patches_dir)

        # [cite_start]特征提取 [cite: 588]
        output_path = Path(features_dir) / f"{slide_id}.h5"
        if output_path.exists():
            print(f"  特征文件已存在: {output_path}")
            with h5py.File(output_path, 'r') as f:
                n_patches = f['features'].shape[0]
            return True, n_patches

        success, n_patches = self.extract_features_from_h5(
            slide_path, h5_path, output_path, batch_size
        )

        # 🟢 优化：仅在单张超大 WSI 的大循环结束时执行一次显存清理，避免小循环频繁调用导致运行效率崩塌
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()

        return success, n_patches


def process_dataset(data_dir, patches_dir, features_dir, checkpoint_path, dataset_name,
                    progress_tracker=None, extractor=None, **kwargs):
    """处理单个数据集"""
    data_dir = Path(data_dir)
    patches_dir = Path(patches_dir) / dataset_name
    features_dir = Path(features_dir) / dataset_name

    patches_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    # [cite_start]递归查找所有SVS文件 [cite: 588]
    svs_files = list(data_dir.glob('**/*.svs'))

    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"SVS文件数: {len(svs_files)}")
    print(f"切片目录: {patches_dir}")
    print(f"特征目录: {features_dir}")
    print(f"{'='*60}")

    if not svs_files:
        print(f"警告: 未找到SVS文件!")
        return [], extractor

    # [cite_start]初始化提取器 [cite: 588]
    if extractor is None:
        extractor = CLAMFeatureExtractor(
            checkpoint_path,
            device=kwargs.get('device', 'cuda'),
            patch_size=kwargs.get('patch_size', 512)
        )

    results = []
    processed = 0

    for idx, svs_file in enumerate(svs_files):
        slide_id = svs_file.stem
        output_path = features_dir / f"{slide_id}.h5"

        if progress_tracker:
            progress_tracker.update_dataset(dataset_name, len(svs_files), processed)

        # [cite_start]跳过已处理 [cite: 588]
        if output_path.exists():
            print(f"\n[{idx+1}/{len(svs_files)}] 跳过: {slide_id}")
            try:
                with h5py.File(output_path, 'r') as f:
                    n_patches = f['features'].shape[0]
            except:
                n_patches = 0

            if progress_tracker:
                progress_tracker.update_slide(dataset_name, slide_id, 'skipped', n_patches)

            results.append({'slide_id': slide_id, 'status': 'skipped', 'n_patches': n_patches})
            processed += 1
            continue

        try:
            print(f"\n[{idx+1}/{len(svs_files)}] 处理: {slide_id}")
            start_time = time.time()

            success, n_patches = extractor.process_slide(
                svs_file, patches_dir, features_dir,
                batch_size=kwargs.get('batch_size', 64)
            )
            elapsed = time.time() - start_time

            if progress_tracker:
                progress_tracker.update_slide(dataset_name, slide_id,
                                              'success' if success else 'failed',
                                              n_patches, elapsed)

            results.append({
                'slide_id': slide_id,
                'status': 'success' if success else 'failed',
                'n_patches': n_patches,
                'time': elapsed
            })
        except Exception as e:
            print(f"  处理失败: {e}")
            if progress_tracker:
                progress_tracker.update_slide(dataset_name, slide_id, 'error', 0)
            results.append({'slide_id': slide_id, 'status': 'error', 'error': str(e)})

        processed += 1

    return results, extractor


def main():
    parser = argparse.ArgumentParser(description='Libra-MIL 特征提取 (CLAM版本)')
    parser.add_argument('--checkpoint', type=str,
                        default='/mnt/sda1/ln_workspace/CONCH/checkpoints/pytorch_model.bin',
                        help='CONCH模型权重')
    parser.add_argument('--output_dir', type=str, default='/mnt/sda2/WSI/muti-modal/TCGA-RCC-fea',
                        help='输出根目录')
    parser.add_argument('--patch_size', type=int, default=512,
                        help='patch大小 (论文: 512)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批处理大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')

    # [cite_start]数据集路径 [cite: 588]
    parser.add_argument('--kich_dir', type=str, default='/mnt/sda2/WSI/TCGA-KICH')
    parser.add_argument('--kirc_dir', type=str, default='/mnt/sda2/WSI/TCGA-KIRC')
    parser.add_argument('--kirp_dir', type=str, default='/mnt/sda2/WSI/TCGA-KIRP')

    parser.add_argument('--progress_file', type=str, default='extraction_progress.json')

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    patches_dir = output_dir / 'patches'
    features_dir = output_dir / 'features'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 进度追踪
    progress_tracker = ProgressTracker(output_dir / args.progress_file)

    # [cite_start]数据集列表 [cite: 588]
    datasets = [
        ('TCGA-KICH', args.kich_dir),
        ('TCGA-KIRC', args.kirc_dir),
        ('TCGA-KIRP', args.kirp_dir)
    ]

    # [cite_start]统计 [cite: 588]
    total_slides = 0
    for name, data_dir in datasets:
        if Path(data_dir).exists():
            n = len(list(Path(data_dir).glob('**/*.svs')))
            total_slides += n
            print(f"  {name}: {n} 个SVS文件")
        else:
            print(f"警告: 路径不存在 -> {data_dir}")

    print(f"\n总共 {total_slides} 个SVS文件")
    progress_tracker.start(total_slides)

    # [cite_start]处理 [cite: 588]
    all_results = {}
    extractor = None

    for name, data_dir in datasets:
        if not Path(data_dir).exists():
            continue
        results, extractor = process_dataset(
            data_dir, patches_dir, features_dir, args.checkpoint, name,
            progress_tracker=progress_tracker,
            extractor=extractor,
            device=args.device,
            patch_size=args.patch_size,
            batch_size=args.batch_size
        )
        all_results[name] = results

    progress_tracker.finish()

    # 保存汇总
    summary_path = output_dir / 'extraction_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("特征提取完成!")
    print(f"输出目录: {output_dir}")
    print(f"切片目录: {patches_dir}")
    print(f"特征目录: {features_dir}")
    print(f"汇总文件: {summary_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

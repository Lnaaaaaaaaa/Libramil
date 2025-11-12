# Libra-MIL: Multimodal Prototypes Stereoscopic Infused with Task-specific Language Priors for Few-shot Whole Slide Image Classification

**Libra-MIL** explores multimodal few-shot learning in computational pathology.  
It fuses patch-level visual embeddings from WSIs with task-specific language priors, enabling interpretable and data-efficient pathology classification.

## Overview

While Large Language Models (LLMs) are promising for pathology, giga-pixel WSIs make full supervision infeasible.  
Libra-MIL addresses this through a multimodal MIL framework that constructs **visual and textual prototypes** for few-shot classification.  
We propose a **stereoscopic optimal transport (SOT)** alignment to enhance cross-modal synergy and interpretability.

## Datasets

Experiments are conducted on three public datasets:

- **[TCGA-RCC](https://portal.gdc.cancer.gov/)**
- **[NSCLC](https://portal.gdc.cancer.gov/)**
- **[CAMELYON16](https://camelyon16.grand-challenge.org/Download/)**

## Pretrained Weights

This project uses pretrained weights from [CONCH](https://huggingface.co/MahmoodLab/CONCH) for the text encoder.

After downloading the pretrained weights, place them in:

```
./conch/pytorch_model.bin
```

No additional trained weights are provided.

## Training & Evaluation

```bash
python main.py \   
--data_split_json ./data/tcga_split.json \
--data_csv ./data/labels.csv \  
--h5_file_dir ./data/features/  \
--instance_path ./text_prompt/TCGA_RCC_instance_prompt.json \
--bag_path ./text_prompt/TCGA_RCC_two_scale_text_prompt.csv \
--text_model_weights_path ./conch/pytorch_model.bin \
--save_dir ./results/   \
--num_vis_prototypes 6  \
--num_classes 2 
```

Training and evaluation are performed in a single run.

## Citation

```
@misc{zhuang2025libramilmultimodalprototypesstereoscopic,
      title={Libra-MIL: Multimodal Prototypes Stereoscopic Infused with Task-specific Language Priors for Few-shot Whole Slide Image Classification}, 
      author={Zhenfeng Zhuang and Fangyu Zhou and Liansheng Wang},
      year={2025},
      eprint={2511.07941},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.07941}, 
}
```




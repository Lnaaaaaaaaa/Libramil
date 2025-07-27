import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import get_scheduler
import json
import pandas as pd

from models.mil_ot import MIL_MultiPrompt_OTFusion
from models.text_encoder import TextEncoder
from utils.general import compute_metrics, CSVWriter, write_summary_log
from datasets.dataset import get_dataloader

def get_config():
    import argparse
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--data_split_json', type=str, default=None, help='JSON file containing data split information')
    parser.add_argument('--data_csv', type=str, default=None, help='CSV file containing data split information')
    parser.add_argument('--h5_file_dir', type=str, default=None, help='directory containing h5 files for WSI patches')
    parser.add_argument('--instance_path', type=str, default=None, help='Path to the instance text file for structural prompts')
    parser.add_argument('--bag_path', type=str, default=None, help='Path to the bag text file for structural prompts')
    parser.add_argument('--save_dir', type=str, default=None)

# model
    parser.add_argument('--feats_dim', type=int, default=512, help='Dimension of the features')
    parser.add_argument('--num_struct_prompts', type=int, default=4, help='Number of structural prompts')
    parser.add_argument('--num_vis_prototypes', type=int, default=4, help='Number of visual prototypes')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for classification')
    parser.add_argument('--pooling_type', type=str, default='attention', choices=['gated_attention', 'attention', 'mean'], help='Pooling type for MIL')
    parser.add_argument('--use_proj', type=bool, default=True, help='Whether to use projection layers')
    parser.add_argument('--ot_epsilon', type=float, default=0.05, help='Epsilon for optimal transport')
    parser.add_argument('--ot_iter', type=int, default=20, help='Number of iterations for optimal transport')

    parser.add_argument('--text_model_weights_path', type=str, default=None)

# train
    parser.add_argument('--flods', type=int, default=5)
    parser.add_argument('--epoches', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=7, help='Random seed for reproducibility')

    args = parser.parse_args()
    return args


def seed_torch(seed=7, device=None):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(model, dataloader, device, epoch, optimizer, lr_scheduler, num_classes):
    model.train()
    y_true, y_pred, y_score = [], [], []
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for feats, labels in loop:
        feats = feats.to(device)
        labels = labels.to(device)

        res_dic = model(feats, labels)
        loss = res_dic['loss']
        logits = res_dic['logits']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        preds = torch.argmax(logits, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())
        y_score.extend(logits.detach().cpu().numpy())

    y_score = np.array(y_score)  
    return compute_metrics(y_true, y_pred, y_score, num_classes)

def evaluate(model, dataloader, device, epoch, num_classes):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for feats, labels in loop:
            feats = feats.to(device)
            labels = labels.to(device)
            res_dic = model(feats, labels)
            logits = res_dic['logits'].detach()

            probs = torch.softmax(logits, dim=1)  
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    y_score = np.array(y_score)
    return compute_metrics(y_true, y_pred, y_score, num_classes)

def main():
    args = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_torch(args.seed, device)

    with open(args.instance_path, "r") as f:
        struct_llm = json.load(f)
    bag_llm = pd.read_csv(args.bag_path, header=None)[0].tolist()[args.num_classes:]

    args.num_struct_prompts = len(struct_llm)

    text_encoder = TextEncoder(args.feats_dim, args.text_model_weights_path)
    T_struct_llm = text_encoder(struct_llm).to(device)
    T_bag_llm = text_encoder(bag_llm).to(device)

    os.makedirs(os.path.join(args.save_dir ,'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir ,'logs'), exist_ok=True)


    final_csv = CSVWriter(filename=os.path.join(args.save_dir ,'logs', 'final_log.csv'), header=[
        'flod', 'test_acc', 'test_auc', 'test_f1_score'
        ])
    
    for fi in range(args.flods):
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

        loaders = get_dataloader(args.data_split_json, args.data_csv, args.h5_file_dir, idx=fi)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        num_training_steps = len(loaders['train']) * args.epoches
        lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)

        best_acc = 0.0
        best_auc = 0.0
        header=[
            'epoch', 'final_epoch', 'train_acc', 'train_auc', 'train_f1_score',
            'valid_acc', 'valid_auc', 'valid_f1_score',
            'test_acc', 'test_auc', 'test_f1_score'
        ]
        log_csv = CSVWriter(filename=os.path.join(args.save_dir ,'logs', f'log_{fi}.csv'), header=header)

        for epoch in range(args.epoches):
            train_acc, train_auc, train_f1_score=train(model, loaders['train'], device, epoch, optimizer, lr_scheduler, args.num_classes)
            valid_acc, valid_auc, valid_f1_score = evaluate(model, loaders['valid'], device, epoch, args.num_classes)
            test_acc, test_auc, test_f1_score = evaluate(model, loaders['test'], device, epoch, args.num_classes)

            if valid_acc > best_acc:
                best_acc = valid_acc
                final_epoch = epoch+1
                torch.save(model.state_dict(), os.path.join(args.save_dir ,f'checkpoints/best_model_{fi}.pt'))
            elif valid_acc == best_acc and valid_auc > best_auc:
                best_auc = valid_auc
                final_epoch = epoch+1
                torch.save(model.state_dict(), os.path.join(args.save_dir ,f'checkpoints/best_model_{fi}.pt'))
            
            log_csv.write_row([epoch+1, final_epoch, train_acc, train_auc, train_f1_score,
                            valid_acc, valid_auc, valid_f1_score,
                            test_acc, test_auc, test_f1_score
                            ])
            
        model.load_state_dict(torch.load(os.path.join(args.save_dir ,f'checkpoints/best_model_{fi}.pt')))
        test_acc, test_auc, test_f1_score = evaluate(model, loaders['test'], device, final_epoch, args.num_classes)
        final_csv.write_row([
            fi, test_acc, test_auc, test_f1_score
        ])
    
    write_summary_log(
        os.path.join(args.save_dir ,'logs', 'final_log.csv'), 
        os.path.join(args.save_dir ,'logs', 'summary_log.csv')
    )

if __name__ == "__main__":
    main()

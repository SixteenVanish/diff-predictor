import torch
import argparse
import numpy as np
from mmengine.device import get_device
from mmpretrain.apis import get_model
from mmengine.config import Config
import itertools

def parse_args():
    parser = argparse.ArgumentParser(description='mytest')
    parser.add_argument('--config', default='configs/byol/byol_resnet18_1xb256-coslr-200e_tinyimagenet_diff-mlp-sgd_v9-2.py')
    parser.add_argument('--checkpoint', default='work_dirs/byol_resnet18_1xb256-coslr-200e_tinyimagenet_diff-mlp-sgd_v9-2/epoch_200.pth')
    args = parser.parse_args()
    return args
        
def main():
    args = parse_args()
    
    cfg = Config.fromfile(args.config)
    model = get_model(cfg, args.checkpoint, device=get_device())
    for param in model.model.parameters():  
        if(param.shape[1]==768):
            print("\nnoisy_latent_part:")
            noisy_latent_part = param[:, :256]
            print(torch.mean(noisy_latent_part))
            
            print("\ncontidion_part:")
            contidion_part = param[:, 256:256*2]
            print(torch.mean(contidion_part))
            
            print("\ntimestep_part:")
            timestep_part = param[:, 256*2:]
            print(torch.mean(timestep_part))
            
def make_pairs0(list1, list2):
    pairs = []
    for idx1, item1 in enumerate(list1):
        for idx2, item2 in enumerate(list2):
            if idx1 == idx2:    # 如果下标相同，则跳过
                continue
            pairs.append([item1, item2])
    return pairs

def make_pairs1(dict1, dict2):
    pairs = []
    for idx1, item1 in dict1.items():
        for idx2, item2 in dict2.items():
            if idx1 == idx2:    # 如果下标相同，则跳过
                continue
            pairs.append([item1, item2])
    return pairs

def make_pairs2(list1, list2):
    pairs = []
    for item1 in list1:
        for item2 in list2:
            if item1[0] == item2[0]:    # 如果下标相同，则跳过
                continue
            pairs.append([item1[1], item2[1]])
    return pairs

def make_pairs3(list1, list2, noSamePos=True):
    pairs = []
    for idx1, item1 in enumerate(list1):
        for idx2, item2 in enumerate(list2):
            if noSamePos:
                if idx1 == idx2:    # 如果下标相同，则跳过
                    continue
            pairs.append([item1, item2])
    return pairs

def cosine_similarity(tensor1, tensor2):
    dot_product = torch.sum(tensor1 * tensor2, dim=-1)
    norm1 = torch.norm(tensor1, dim=-1)
    norm2 = torch.norm(tensor2, dim=-1)
    similarity = dot_product / (norm1 * norm2 + 1e-8)  # 避免除以0
    return similarity

if __name__ == '__main__':
    
    beta_prod_t = torch.arange(1,9)
    model_output = torch.ones((8,3,2))
    sample = torch.ones((8,3,2))
    
    print(beta_prod_t.shape)
    print(model_output.shape)
    print(sample.shape)
    
    print("tmp1")
    tmp1 = torch.einsum("b,bcd->bcd", beta_prod_t ** (0.5), model_output)
    print(tmp1)
    print(tmp1.shape)
    tmp2 = sample - tmp1
    
    print("tmp2")
    print(tmp2)
    print(tmp2.shape)
    
    alpha_prod_t = torch.arange(1,9)
    alpha_prod_t = alpha_prod_t.unsqueeze(1).unsqueeze(1)
    print(alpha_prod_t.shape)
    tmp3 = tmp2 / ( alpha_prod_t ** (0.5))
    print(tmp3)
    print(tmp3.shape)
    # pred_original_sample = (sample - torch.einsum("b,bcd->bcd", beta_prod_t ** (0.5), model_output)) / (
    #                 alpha_prod_t ** (0.5)).view(-1, 1)
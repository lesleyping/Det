from data import *
from ssd import build_ssd

import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data

parser = argparse.ArgumentParser(description='SSD with torch')
parser.add_argument('--dataset', default='VOC', type=str)
parser.add_argument('--dataset_root', default=VOC_ROOT)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
args = parser.parse_args()

def train():
    # load data
    if args.dataset == 'VOC':
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=BaseTransform(cfg['min_dim'], 
                                                      MEANS))
    
    #ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])

    data_loader = data.DataLoader(dataset, batch_size=2,
                                  num_workers=args.num_workers,
                                  shuffle=True,
                                  collate_fn=detection_collate,
                                  pin_memory=True)
    
    for images, targets in data_loader:
        print(images.shape)
    
if __name__ == "__main__":
    train()
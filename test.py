import os
import math
import random
import argparse
import numpy as np
from tqdm import tqdm
import cv2
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modules import define_network
from losses import Criterion
from utilities.checkpoint import Saver
from utilities.data_utils import show_img, ensure_dir

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def tester(args, data_loader_test):
    """ -------------------- build net -------------------- """
    net = define_network(args.arch)()
    net = nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(args.net_path), strict=False)
    print("Load pretrained net from " + args.net_path)

    params_count = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Network parameters: ", params_count)

    net = net.eval()

    delta_time = 0
    """ -------------------- start testing -------------------- """
    tbar = tqdm(data_loader_test)
    for idx, (short_0, short_1, long_0, long_1, relong_0, relong_1, true_0, true_1, evg_0, evg_1, evg_2, prefix) in enumerate(tbar):
        """ -------------------- load to GPU -------------------- """
        short = short_0.cuda()
        long = long_1.cuda()
        evg_0 = evg_0.cuda()
        evg_1 = evg_1.cuda()
        evg_2 = evg_2.cuda()
        true = true_0.cuda()
        
        event1 = torch.cat((evg_0, evg_1), 1)
        start = time.time()
        preds, _ = net(short, long, event1, evg_2)
        end = time.time()
        delta_time += end - start

        seq_name = prefix[0].split('/')[-2]
        bag_name = prefix[0].split('/')[-1].split('.')[0]

        ensure_dir(os.path.join('./test', args.exp_name, args.dataset_name, seq_name, 'Pred_s'))
        cv2.imwrite(os.path.join('./test', args.exp_name, args.dataset_name, seq_name, 'Pred_s', bag_name+'.png'), show_img(preds[-1]))

        ensure_dir(os.path.join('./test', args.exp_name, args.dataset_name, seq_name, 'Pred_l'))
        cv2.imwrite(os.path.join('./test', args.exp_name, args.dataset_name, seq_name, 'Pred_l', bag_name+'.png'), show_img(preds[-2]))

        ensure_dir(os.path.join('./test', args.exp_name, args.dataset_name, seq_name, 'True'))
        cv2.imwrite(os.path.join('./test', args.exp_name, args.dataset_name, seq_name, 'True', bag_name+'.png'), show_img(true))
        
        torch.cuda.empty_cache()
    print("Average time per image: {:.4f} seconds".format(delta_time / len(data_loader_test)))
    return 0

@torch.no_grad()
def main(args):
    """ -------------------- load dataset -------------------- """
    if args.dataset_name == 'REDS':
        from dataloader_reds import Dataloader_test
    elif args.dataset_name == 'RLIED':
        from dataloader_pied import Dataloader_test
    dataset_test = Dataloader_test(args)
    data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    """ -------------------- Start testing -------------------- """
    print("---------- Network Architecture:", args.arch, "----------")
    tester(args, data_loader_test)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test EDEI")
    parser.add_argument("--dataset_path", type=str, default="", help="data path")
    parser.add_argument("--dataset_name", type=str, default="REDS", help="data path")

    parser.add_argument("--arch", type=str, default="UNet_de_ca_denc_ddec")

    parser.add_argument("--net_path", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="")
    
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=2, type=int)

    parser.add_argument("--description", type=str, default='test on reds dataset')
    parser.add_argument("--seed", default=3, type=int)

    args = parser.parse_args()
    main(args)

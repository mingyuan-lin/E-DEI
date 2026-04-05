import os
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modules import define_network
from losses import Criterion
from utilities.checkpoint import Saver

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def trainer(args, data_loader_train, checkpoint_saver):
    """ -------------------- build criterion -------------------- """
    print(args.loss_function, ':', args.loss_weight)
    criterion = Criterion(args)

    """ -------------------- build Net -------------------- """
    net = define_network(args.arch)()
    net = nn.DataParallel(net).cuda()
    net = net.train()

    if args.is_resume:
        net.load_state_dict(torch.load(args.net_path), strict=False)
        print("Load pretrained net from " + args.net_path)

    param_dicts = [
         {"params": net.parameters(),
         "lr": args.lr_net,
         },
    ]
    optimizer = torch.optim.Adam(param_dicts)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch)

    if args.is_resume:
        for i in range(0, args.start_epoch):
            lr_scheduler.step()

    iter = 0
    """ -------------------- start training -------------------- """
    for epoch in range(args.start_epoch, args.num_epoch):
        print("Epoch: %d" % epoch)
        print("current learning rate", lr_scheduler.get_last_lr())
        
        loss_sum = 0.
        loss_gt = 0.

        tbar = tqdm(data_loader_train)
        for idx, (short_0, short_1, long_0, long_1, relong_0, relong_1, true_0, true_1, evg_0, evg_1, evg_2, prefix) in enumerate(tbar):
            """ -------------------- load to GPU -------------------- """
            short = short_0.cuda()
            long = long_1.cuda()
            evg_0 = evg_0.cuda()
            evg_1 = evg_1.cuda()
            evg_2 = evg_2.cuda()
            true = true_0.cuda()
            
            event1 = torch.cat((evg_0, evg_1), 1)
            preds, _ = net(short, long, event1, evg_2)

            loss, loss_list = criterion(preds, true)
            loss_sum += loss.item()
            loss_gt += loss_list[0].item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter % 500 == 0:
                print("l_sum: %.4f, loss_gt: %.4f" % (loss_sum/(idx+1), loss_gt/(idx+1)))
                checkpoint_saver.save_checkpoint(net, 'epoch_' + str(epoch) + '_model.pth')
            iter += 1

        lr_scheduler.step()
        torch.cuda.empty_cache()
        print("l_sum: %.4f, loss_gt: %.4f" % (loss_sum/(idx+1), loss_gt/(idx+1)))
        checkpoint_saver.save_checkpoint(net, 'epoch_' + str(epoch) + '_model.pth')

    return 0


def main(args):
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    """ -------------------- build checkpoint -------------------- """
    checkpoint_saver = Saver(args)
    
    """ -------------------- load dataset -------------------- """
    if args.dataset_name == 'REDS':
        from dataloader_reds import Dataloader_train
    elif args.dataset_name == 'RLIED':
        from dataloader_pied import Dataloader_train
    dataset_train = Dataloader_train(args)
    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    """ -------------------- Start training -------------------- """
    print("---------- Network Architecture:", args.arch, "----------")
    trainer(args, data_loader_train, checkpoint_saver)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EDEI")
    parser.add_argument("--dataset_path", type=str, default="", help="data path")
    parser.add_argument("--dataset_name", type=str, default="REDS", help="data path")
    parser.add_argument("--model_path", type=str, default="./run/", help="model saving path")

    parser.add_argument("--arch", type=str, default="UNet_de_ca_denc_ddec")

    parser.add_argument("--is_resume", action="store_true", default=False)
    parser.add_argument("--net_path", type=str, default="", help="pretrained model path")
    
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--lr_net", default=1e-04, type=float)
    
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--num_epoch", default=100, type=int)
        
    parser.add_argument("--loss_function", type=str, default="[loss_l1]")
    parser.add_argument("--loss_weight", type=list, default=[1])

    parser.add_argument("--crop_height", type=int, help="input image height", default=256)
    parser.add_argument("--crop_width", type=int, help="input image width", default=256)

    parser.add_argument("--description", type=str, default='train WITH de & ca & denc & ddec & dp')
    parser.add_argument("--seed", default=31, type=int)

    args = parser.parse_args()
    main(args)

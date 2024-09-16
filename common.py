from typing import Any, Callable, Optional
from torch.utils.data import DataLoader
import torch.nn as nn

import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree

import torch
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from ImprovedENet import ImprovedENet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)

from losses import (CrossEntropy)

datasets_params = {
    "TOY2": {'K': 2, 'net': None, 'B': 2},  # Replace None with the appropriate network for TOY2
    "SEGTHOR": {'K': 5, 'net': ImprovedENet, 'B': 8}
}


def setup(args, custom_setup: Optional[Callable] = None) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    

    if custom_setup:
        return custom_setup(args)
    else:
        # ... (existing setup code)
        # Networks and scheduler
        gpu: bool = args.gpu and torch.cuda.is_available()
        device = torch.device("cuda") if gpu else torch.device("cpu")
        print(f">> Picked {device} to run experiments")

        K: int = datasets_params[args.dataset]['K']

        
        net = datasets_params[args.dataset]['net'](1, K)
        net.init_weights()
        net.to(device)

        lr = 0.0005
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

        # check if the network has an init_weights method before calling it
        if hasattr(net, 'init_weights'):
            net.init_weights()

        # Dataset part
        B: int = datasets_params[args.dataset]['B']
        root_dir = Path("data") / args.dataset

        img_transform = transforms.Compose([
            lambda img: img.convert('L'),
            lambda img: np.array(img)[np.newaxis, ...],
            lambda nd: nd / 255,  # max <= 1
            lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])

        gt_transform = transforms.Compose([
            lambda img: np.array(img)[...],
            # The idea is that the classes are mapped to {0, 255} for binary cases
            # {0, 85, 170, 255} for 4 classes
            # {0, 51, 102, 153, 204, 255} for 6 classes
            # Very sketchy but that works here and that simplifies visualization
            lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,  # max <= 1
            lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
            lambda t: class2one_hot(t, K=K),
            itemgetter(0)
        ])

        train_set = SliceDataset('train',
                                root_dir,
                                img_transform=img_transform,
                                gt_transform=gt_transform,
                                debug=args.debug)
        train_loader = DataLoader(train_set,
                                batch_size=B,
                                num_workers=5,
                                shuffle=True)

        val_set = SliceDataset('val',
                            root_dir,
                            img_transform=img_transform,
                            gt_transform=gt_transform,
                            debug=args.debug)
        val_loader = DataLoader(val_set,
                                batch_size=B,
                                num_workers=5,
                                shuffle=False)

        args.dest.mkdir(parents=True, exist_ok=True)
        
        return net, optimizer, device, train_loader, val_loader, K



    #return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args, custom_setup: Optional[Callable] = None):

    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args, custom_setup)


    if args.mode == "full":
        loss_fn = CrossEntropy(idk=list(range(K)))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset in ['SEGTHOR', 'SEGTHOR_STUDENTS']:
        loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)
    
    

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0

    for e in range(args.epochs):
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()

                    if m == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            print(f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC")
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", 'w') as f:
                    f.write(str(e))

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                    rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")

    return best_dice

import os
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from time import time_ns

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from densenet import densenet121
from dataset import load_dataset
from state import save_checkpoint, load_checkpoint


def train(model, optimizer, train_loader, scaler):
    model.train()
    for (x, y) in train_loader:
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=True):
            out = model(x)
            out = F.log_softmax(out, dim=1)
            loss = F.nll_loss(out, y)
        scaler.scale(loss).backward()         # put this line inside the decorator if gradient checkpointing is used
        scaler.step(optimizer)
        scaler.update()


def test(model, val_loader):
    model.eval()
    correct = 0.
    with tqdm(total=len(val_loader.dataset)) as progress_bar:
        with torch.no_grad():
            for (x, y) in val_loader:
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)

                out = model(x)
                pred = F.log_softmax(out, dim=1).max(1)[1]

                correct += pred.eq(y).cpu().sum().item()
                progress_bar.update(x.size(0))
    accuracy = (100. * correct) / len(val_loader.dataset)
    return accuracy


def worker(device_id, args):
    rank_id = args.nr * args.gpus + device_id
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank_id
    )
    torch.cuda.set_device(device_id)


    train_set, val_set = load_dataset(args.dataset, args.dataroot)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set,
        num_replicas=args.world_size,
        rank=rank_id,
        shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size,
                                               shuffle=False, num_workers=3, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = densenet121(pretrained=True, num_classes=args.num_classes, memory_efficient=True).cuda(device_id)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id])
    state = load_checkpoint(args.cp_file, device_id, model, optimizer)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    cudnn.benchmark = True

    start_epoch = state.epoch + 1
    for epoch in range(start_epoch, args.max_epochs):
        t0 = time_ns()

        train(model, optimizer, train_loader, scaler)

        t1 = time_ns()
        delta = (t1 - t0) / (10 ** 9)
        print(f"Device {device_id} - Train time: {delta} sec")

        if device_id == 0:
            accuracy = test(model, val_loader)
            print(f"Accuracy: {accuracy}%")

        if epoch in [int(args.max_epochs * 0.5), int(args.max_epochs * 0.75)]:
            optimizer.param_groups[0]['lr'] /= 10.

        if epoch % args.save_interval == 0 and device_id == 0:
            save_checkpoint(state, args.cp_file)

        state.epoch = epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DDP training')
    parser.add_argument('--dataset',
                        help='dataset',
                        type=str,
                        default='CIFAR10')
    parser.add_argument('--dataroot',
                        help='dataroot',
                        type=str,
                        default='./data')
    parser.add_argument('--batch_size',
                        help='total batch size',
                        type=int,
                        default=64)
    parser.add_argument('--max_epochs',
                        help='maximum number of training epoches.',
                        type=int,
                        default=200)
    parser.add_argument('--save_interval',
                        help='save interval in epochs',
                        type=int,
                        default=10)
    parser.add_argument('--lr',
                        help='lr.',
                        type=float,
                        default=0.1)
    parser.add_argument('--num_classes',
                        help='number of classes.',
                        type=int,
                        default=10)
    parser.add_argument('--cp_file',
                        help='checkpoint file',
                        type=str,
                        default='./checkpoints/CIFAR10_mp.pt')

    parser.add_argument('-n', '--nodes',
                        default=1,
                        type=int,
                        metavar='N')
    parser.add_argument('-g', '--gpus',
                        default=2,
                        type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr',
                        default=0,
                        type=int,
                        help='ranking within the nodes')

    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    args.batch_size = args.batch_size // args.gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8080'
    mp.spawn(worker, nprocs=args.gpus, args=(args,))

import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from time import time_ns
from tqdm import tqdm
import torch.optim as optim

from dataset import load_dataset
from densenet import densenet121
from state import save_checkpoint, load_checkpoint


def train(model, optimizer, train_loader):
    model.train()
    for (x, y) in train_loader:
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        optimizer.zero_grad()

        out = model(x)
        out = F.log_softmax(out, dim=1)
        loss = F.nll_loss(out, y)

        loss.backward()
        optimizer.step()


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


def main(args):
    train_set, val_set = load_dataset(args.dataset, args.dataroot)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size,
                                               shuffle=True, num_workers=3, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = densenet121(pretrained=True, num_classes=args.num_classes, memory_efficient=True).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    state = load_checkpoint(args.cp_file, 0, model, optimizer)

    cudnn.benchmark = True

    start_epoch = state.epoch + 1
    for epoch in range(start_epoch, args.max_epochs):
        train(model, optimizer, train_loader)

        accuracy = test(model, val_loader)
        print(f"Accuracy: {accuracy}%")

        if epoch in [int(args.max_epochs * 0.5), int(args.max_epochs * 0.75)]:
            optimizer.param_groups[0]['lr'] /= 10.

        state.epoch = epoch
        if epoch % args.save_interval == 0:
            save_checkpoint(state, args.cp_file)


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
                        help='number of images in a mini-batch per GPU.',
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
                        default='./checkpoints/CIFAR10_base.pt')

    args = parser.parse_args()
    main(args)

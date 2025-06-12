'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models # 사용자 정의 모델이 있는 폴더 및 파일

# utils 폴더에 Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig가 있다고 가정합니다.
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20', # 원본은 resnet18이나 resnet20으로 변경
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet20)') # resnet18 -> resnet20
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()} # 학습률 로깅에 사용됨

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=True, transform=transform_test) # download=True로 변경 (일반적으로 테스트셋도 다운로드)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'): # 'resnet'으로 끝나는 모든 경우 (예: 'resnet20')
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth, # ResNet 계열은 depth를 사용 (models/cifar/resnet.py 참조)
                    block_name=args.block_name,
                )
    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda() # GPU 사용 시 DataParallel로 감싸고 .cuda() 호출
    else:
        # CPU 사용 시 DataParallel 없이 모델 그대로 사용 (필요에 따라)
        # DataParallel은 여러 GPU 사용시 유용. 단일 GPU나 CPU에서는 필수는 아님.
        pass # 특별한 처리 없음

    cudnn.benchmark = True # CUDA 연산 가속화 (입력 크기가 동일할 때 유용)
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = args.dataset + '-' + args.arch # 로그 타이틀 (예: cifar10-resnet20)
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        # DataParallel 사용 여부에 따라 state_dict 로드 방식 조정 필요할 수 있음
        # 만약 저장된 모델이 DataParallel 래핑 없이 저장되었다면, 현재 model(DataParallel 래핑된)에 로드할 때 주의
        # 여기서는 저장/로드 시 모두 DataParallel 래핑을 가정
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        # <<< VRAM 항목 추가 >>>
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Forward/Backward Time (sec)', 'Avg Reserved VRAM (MB)', 'Total Time (sec).'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    total_time = 0
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        current_lr = optimizer.param_groups[0]['lr'] # state['lr'] 대신 optimizer에서 직접 가져옴
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, current_lr))
        
        epoch_start_time = time.time()
        # <<< train 함수에서 avg_vram_train 반환 받도록 수정 >>>
        train_loss, train_acc, fb_time, avg_vram_train = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda) # test 함수는 VRAM 로깅 안 함
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_time += epoch_duration
        
        # <<< logger append에 VRAM 정보 추가 >>>
        logger.append([current_lr, train_loss, test_loss, train_acc, test_acc, fb_time, avg_vram_train, total_time])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    # logger.plot() # utils.Logger에 plot 기능이 있다면 호출
    # savefig(os.path.join(args.checkpoint, 'log.eps')) # utils.savefig가 있다면 호출

    print('Best acc:')
    print(best_acc)
    
    total_minutes, total_seconds = divmod(total_time, 60)
    total_time_str = f"Total training time across all epochs: {int(total_minutes)} minutes and {int(total_seconds)} seconds"
    print(total_time_str)

    # 총 시간 및 최고 정확도 로그 파일에 추가 기록
    with open(os.path.join(args.checkpoint, 'log.txt'), 'a') as log_file:
        log_file.write('\n' + total_time_str + '\n')
        log_file.write(f"Best Accuracy: {best_acc:.2f}%\n")


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    forward_backward_time = AverageMeter() # 각 배치의 순전파+역전파 시간 측정용
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    vram_meter = AverageMeter() # <<< VRAM 사용량 기록용 AverageMeter 추가 >>>

    end = time.time()
    
    bar = Bar('Processing', max=len(trainloader))
    epoch_fb_time_sum = 0 # 에폭 전체의 순전파/역전파 시간 합계

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        # PyTorch 0.4.0 이상에서는 torch.autograd.Variable을 명시적으로 사용할 필요 없음
        # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        batch_fb_start_time = time.time() # 배치별 순전파/역전파 시간 측정 시작
        
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0)) # .item()으로 스칼라 값 추출
        top5.update(prec5.item(), inputs.size(0)) # .item()으로 스칼라 값 추출

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_fb_time = time.time() - batch_fb_start_time # 현재 배치의 순전파/역전파 시간
        forward_backward_time.update(batch_fb_time) # AverageMeter에 기록
        epoch_fb_time_sum += batch_fb_time # 에폭 전체 합계에 더함

        # <<< VRAM 사용량 측정 (MB 단위) >>>
        if use_cuda:
            # 현재 PyTorch 텐서에 의해 할당된 GPU 메모리
            current_vram_reserved = torch.cuda.memory_reserved() / (1024 * 1024) # torch.cuda.memory_allocated() 대신 사용
            vram_meter.update(current_vram_reserved)

        # measure elapsed time for the batch
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        vram_avg_str = f'{vram_meter.avg:.1f}MB' if use_cuda and vram_meter.count > 0 else 'N/A'
        bar.suffix  = ('({batch}/{size}) Data: {data:.3f}s | F/B: {fb:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | '
                       'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f} | VRAM: {vram}').format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    fb=forward_backward_time.avg, # 배치당 평균 F/B 시간 표시
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    vram=vram_avg_str # <<< VRAM 정보 표시 >>>
                    )
        bar.next()
    bar.finish()
    
    avg_vram_epoch = vram_meter.avg if use_cuda and vram_meter.count > 0 else 0
    # <<< epoch_fb_time_sum (에폭 전체 F/B 시간 합계)과 avg_vram_epoch 반환 >>>
    return (losses.avg, top1.avg, epoch_fb_time_sum, avg_vram_epoch)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc # 사용되지 않으므로 제거해도 무방

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    with torch.no_grad(): # PyTorch 0.4.0 이상에서는 volatile=True 대신 사용
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets) # Deprecated

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state_data, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'): # 파라미터 이름 명확화
    filepath = os.path.join(checkpoint, filename)
    torch.save(state_data, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    # global state # state 딕셔너리의 'lr'을 직접 수정하는 대신, optimizer의 param_groups를 통해 lr을 가져오고 설정
    current_lr = optimizer.param_groups[0]['lr'] # 현재 lr 가져오기
    if epoch in args.schedule:
        new_lr = current_lr * args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        # state['lr'] = new_lr # state 딕셔너리도 업데이트 (기존 코드 호환성 또는 다른 곳 참조 시)
        # 하지만 main 루프에서 current_lr = optimizer.param_groups[0]['lr'] 로 직접 가져오므로 필수는 아님

if __name__ == '__main__':
    main()
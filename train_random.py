from __future__ import print_function

import os
import time
import numpy as np
import random
import shutil

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim

from fitting_random_labels.cifar10_data import CIFAR10RandomLabels

from fitting_random_labels import cmd_args
from fitting_random_labels import model_mlp, model_wideresnet

import models.cifar as models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
use_cuda = torch.cuda.is_available()




def get_data_loaders(args, shuffle_train=True):
  if args.data == 'cifar10':
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.data_augmentation:
      transform_train = transforms.Compose([
          transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,
          ])
    else:
      transform_train = transforms.Compose([
          transforms.ToTensor(),
          normalize,
          ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        CIFAR10RandomLabels(root='./data', train=True, download=True,
                            transform=transform_train, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob),
        batch_size=args.batch_size, shuffle=shuffle_train, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        CIFAR10RandomLabels(root='./data', train=False,
                            transform=transform_test, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader
  else:
    raise Exception('Unsupported dataset: {0}'.format(args.data))


# def get_model(args):
#   # create model
#   if args.arch == 'wide-resnet':
#     model = model_wideresnet.WideResNet(args.wrn_depth, args.num_classes,
#                                         args.wrn_widen_factor,
#                                         drop_rate=args.wrn_droprate)
#   elif args.arch == 'mlp':
#     n_units = [int(x) for x in args.mlp_spec.split('x')] # hidden dims
#     n_units.append(args.num_classes)  # output dim
#     n_units.insert(0, 32*32*3)        # input dim
#     model = model_mlp.MLP(n_units)

#   # for training on multiple GPUs.
#   # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
#   # model = torch.nn.DataParallel(model).cuda()
#   model = model.cuda()

#   return model

# cifar 10

def get_model(args):
  # num_classes = 10  

  model = models.__dict__[args.arch](num_classes=args.num_classes)
  model = torch.nn.DataParallel(model).cuda()

  print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

  return model


def train_model(args, model, train_loader, val_loader,logger,
                start_epoch=None, epochs=None):
  cudnn.benchmark = True

  best_acc = 0  # best test accuracy

  # define loss function (criterion) and pptimizer
  criterion = nn.CrossEntropyLoss().cuda()
  optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

  start_epoch = start_epoch or 0
  epochs = epochs or args.epochs

  for epoch in range(start_epoch, epochs):
    adjust_learning_rate(optimizer, epoch, args)

    # train for one epoch
    train_loss, train_acc = train_epoch(train_loader, model, criterion, optimizer, epoch, args)

    # evaluate on validation set
    test_loss, val_prec1 = validate_epoch(val_loader, model, criterion, epoch, args)

    if args.eval_full_trainset:
      tr_loss, test_acc = validate_epoch(train_loader, model, criterion, epoch, args)

    # append logger file
    logger.append( [train_loss, test_loss, train_acc, test_acc])

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
  logger.plot()
  savefig(os.path.join(args.checkpoint, 'log.eps'))

  print('Best acc:')
  print(best_acc)


def train_epoch(train_loader, model, criterion, optimizer, epoch, args):
  """Train for one epoch on the training set"""
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()

  batch_time = AverageMeter()
  data_time = AverageMeter()

  end = time.time()
  # switch to train mode
  model.train()

  bar = Bar('Processing', max=len(train_loader))
  for batch_idx, (input, target) in enumerate(train_loader):
    data_time.update(time.time() - end)

    target = target.cuda(async=True)
    input = input.cuda()
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)

    # compute output
    output = model(input_var)
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1 = accuracy(output.data, target, topk=(1,))[0]
    losses.update(loss.data[0], input.size(0))
    top1.update(prec1[0], input.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()


    # plot progress
    bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    )
    bar.next()
    bar.finish()

  return losses.avg, top1.avg


def validate_epoch(val_loader, model, criterion, epoch, args):
  """Perform validation on the validation set"""
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()

  batch_time = AverageMeter()
  data_time = AverageMeter()

  # switch to evaluate mode
  model.eval()

  end = time.time()
  bar = Bar('Processing', max=len(val_loader))
  for batch_idx, (input, target) in enumerate(val_loader):
    data_time.update(time.time() - end)

    target = target.cuda(async=True)
    input = input.cuda()
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)

    # compute output
    output = model(input_var)
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1 = accuracy(output.data, target, topk=(1,))[0]
    losses.update(loss.data[0], input.size(0))
    top1.update(prec1[0], input.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    # plot progress
    bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    )
    bar.next()
    bar.finish()

  return losses.avg, top1.avg


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
  """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
  lr = args.learning_rate * (0.1 ** (epoch // 30)) * (0.1 ** (epoch // 60))
  for param_group in optimizer.param_groups:
      param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0)
      res.append(correct_k.mul_(100.0 / batch_size))
  return res


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def main():
  
  model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

  args = cmd_args.parse_args()
  state = {k: v for k, v in args._get_kwargs()}

  # setup_logging(args)
  if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

  title = 'cifar-10-' + args.arch
  logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
  logger.set_names(['Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

  # Random seed
  if args.manualSeed is None:
      args.manualSeed = random.randint(1, 10000)
  random.seed(args.manualSeed)
  torch.manual_seed(args.manualSeed)
  if use_cuda:
      torch.cuda.manual_seed_all(args.manualSeed)

  if args.command == 'train':
    train_loader, val_loader = get_data_loaders(args, shuffle_train=True)
    model = get_model(args)

    # logging.info('Number of parameters: %d', sum([p.data.nelement() for p in model.parameters()]))
    train_model(args, model, train_loader, val_loader, logger)


if __name__ == '__main__':
  main()


from __future__ import absolute_import
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

title = 'cifar-10-' + 'att_resnet-110'

log_path = '/media/jaden/DeepLearningCode/pytorch-classification/checkpoints/cifar100/att_resnet-110/log.txt'

logger = Logger(log_path, title=title)
logger.plot()

plt.show()
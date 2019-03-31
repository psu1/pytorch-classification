# A simple torch style logger
# (C) Wei YANG 2017
from __future__ import absolute_import
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
# plt.switch_backend('agg')

__all__ = ['Logger', 'LoggerMonitor', 'savefig']

def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)
    
def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        # np.asarray(numbers[name]) is a str list, convert to float array
        plt.plot(x, np.asarray(numbers[name]).astype(np.float) ) 
    return [logger.title + '(' + name + ')' for name in names]

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                # print('name: ', name)
                self.names = name.rstrip().split('\t')
                # print('self.names: ', self.names)
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    # print('numbers: ', numbers)
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):   
        names = self.names if names == None else names
        numbers = self.numbers
        plt.figure(figsize=(18,6))
        plt.subplot(121)
        for _, name in enumerate(names[1:3]):
            # print('namme {} data {} type {}'  .format(name, numbers[name], type(numbers[name])))
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names[1:3]])

        plt.subplot(122)
        for _, name in enumerate(names[3:]):
            # print('namme {} data {} array {} type {}'  .format(name, numbers[name],np.asarray(numbers[name]), type(numbers[name])))
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names[3:]])

        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()

class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure(figsize=(18,6))
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)             
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlabel('Epoch ')
        plt.ylabel('Acc % ')
        plt.grid(True)
        
                    
if __name__ == '__main__':

    paths1 = {
    'resnet-110' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoints/cifar100/resnet-110/log.txt',
    'alexnet' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoints/cifar100/alexnet/log.txt',
    'resnet-26' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoints/cifar100/resnet-26/log.txt',
    'vgg11' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoints/cifar100/vgg11/log.txt',
    'vgg19_bn_q' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoints/cifar100/vgg19_bn_q/log.txt',
    'densenet-bc-100-12' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoints/cifar100/densenet-bc-100-12/log.txt',
    }
  

    paths2 = {
    'dropout-0.0' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoints/cifar100/resnet-110/log.txt',
    'dropout-0.1' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoint/cifar100/dropout/resnet_dropout1-110/log.txt',
    'dropout-0.3' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoint/cifar100/dropout/resnet_dropout3-110/log.txt',
    'dropout-0.5' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoint/cifar100/dropout/resnet_dropout5-110/log.txt',
    'dropout-0.7' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoint/cifar100/dropout/resnet_dropout7-110/log.txt',
    'dropout-0.9' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoint/cifar100/dropout/resnet_dropout9-110/log.txt',
    'dropout2d-0.1' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoint/cifar100/resnet_dropout2D1-110/log.txt',
    'dropout2d-0.3' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoint/cifar100/resnet_dropout2D3-110/log.txt',
    # 'dropout-1' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoint/cifar100/resnet_dropout10-110/log.txt',
    }

    paths3 = {
    'resnet-110' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoints/cifar100/resnet-110/log.txt',
    'resnet-110-minus' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoints/cifar100/resnet-110-minus/log.txt',
    'resnet-110_dropout-out' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoints/cifar100/resnet_dropout-out-110/log.txt',
    # 'resnet-110' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoints/cifar100/resnet-110/log.txt',
    }

    paths4 = {
    'resnet-110' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoints/cifar100/resnet-110/log.txt',
    'resnet-110-spatialAtt-wo-bn' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoints/cifar100/resnet-110-spatialAtt-wo-bn/log.txt',
    'resnet-110-chanelAtt_new' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoints/cifar100/resnet-110-chanelAtt_new/log.txt',
    # 'resnet-110' : '/media/jaden/DeepLearningCode/pytorch-classification/checkpoints/cifar100/resnet-110/log.txt',
    }


    # fields = [['Valid Acc.']]
    # FULL comparison
    fields = [['Train Loss'], ['Valid Loss'],  ['Train Acc.'], ['Valid Acc.']]
    monitor = LoggerMonitor(paths4)

    for field in fields:  
        print field    
        monitor.plot(names=field)
        # plt.show()
        savefig('cifar100_all' + field[0] + '.eps')

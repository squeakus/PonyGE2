#!/usr/bin/env python
"""

@author: XiaofanXu and Jonathan Byrne
@17/01/18 11:09

"""

from fitness.base_ff_classes.base_ff import base_ff
import random
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
from torch.autograd import Variable

import time
import shutil

print_freq = 10
max_epoch = 200
# 0.1 0.001
base_lr = 0.01
# 0.1 0.9
momentum = 0.9
weight_decay = 0.0005

class create_network(nn.Module, base_ff):

    def __init__(self, settings, input_shape=(3, 120, 160)):
        super(create_network, self).__init__()

        self.name = ""
        self.conv = nn.ModuleList()

        for i in range(len(settings['conv_layers'])):
            k, n_o = settings['conv_layers'][i]
            self.name += "C_{}_{}_".format(k, n_o)

            if i == 0:
                self.conv.append(nn.Conv2d(3, n_o, kernel_size=k))
                self.conv.append(nn.MaxPool2d(2))
                self.conv.append(nn.ReLU(True))

            else:
                self.conv.append(nn.Conv2d(n_i, n_o, kernel_size=k))
                self.conv.append(nn.MaxPool2d(2))
                self.conv.append(nn.ReLU(True))

            n_i = n_o

        self.n_size = self._get_conv_output(input_shape)
        print("size", self.n_size)

        if settings['fc'] == 0:

            self.feat_position = nn.Linear(self.n_size, 2)
            self.softmax = nn.Softmax()
            self.name += "2"

        else:
            self.fc = nn.ModuleList()
            for i in range(settings['fc']):
                fc_o = settings['fc_layers'][i]
                self.name += "FC_{}_".format(fc_o)

                if i == 0:
                    self.fc.append(nn.Linear(self.n_size, fc_o))
                    self.fc.append(nn.ReLU(True))
                else:
                    self.fc.append(nn.Linear(fc_i, fc_o))
                    self.fc.append(nn.ReLU(True))

                fc_i = fc_o
            self.feat_position = nn.Linear(fc_o, 2)
            self.softmax = nn.Softmax()
            self.name += "2"

    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):

        for i in range(len(self.conv)):
            x = self.conv[i](x)
        return x

    def forward(self, x, settings):

        for i in range(len(self.conv)):
            x = self.conv[i](x)

        out = x.view(x.size(0), self.n_size)

        if settings['fc'] == 0:

            out = self.feat_position(out)
            out = self.softmax(out)
        else:
            for i in range(len(self.fc)):
                out = self.fc[i](out)

            out = self.feat_position(out)
            out = self.softmax(out)

        return out, self.name


class dnn_fitness(base_ff):
    maximise = True
    multi_objective = True

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        test_set = face(['./deeplearn/test.hdf5'])
        train_set = face(['./deeplearn/train.hdf5'])
        self.test_loader = DataLoader(test_set, shuffle=False, batch_size=32,
                                      num_workers=1)

        self.train_loader = DataLoader(train_set, batch_size=32, shuffle=False,
                                       num_workers=1, pin_memory=True)
        # Set list of individual fitness functions.
        self.num_obj = 2
        dummyfit = base_ff()
        dummyfit.maximise = True
        self.fitness_functions = [dummyfit, dummyfit]
        self.default_fitness = [float('nan'), float('nan')]


    def evaluate(self, ind, **kwargs):
        phenotype = ind.phenotype
        fitness = 0
        settings = {}

        try:
            t0 = time.time()
            exec(phenotype, settings)
            model = create_network(settings)
            print(model)
            if torch.cuda.is_available():
                model.cuda()

        except Exception as e:
            fitness = self.default_fitness
            print("Error", e)

        size = 0
        for key, module in model._modules.items():
            params = sum([np.prod(p.size()) for p in module.parameters()])
            size += params

        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(
            model.parameters(), base_lr, momentum=momentum, weight_decay=weight_decay)
        best_prec1 = 0
        for epoch in range(1, max_epoch):

            # train for one epoch
            train(self.train_loader, model, criterion,
                  optimizer, epoch, settings)

            # evaluate on validation set
            prec1, val_loss, name = test(
                self.test_loader, criterion, model, settings)

            # remember best prec@1 and save checkpoint
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'best_prec1': best_prec1,
                             'optimizer': optimizer.state_dict(),
                             }, filename='deeplearn/model_{}.pth.tar'.format(name))
        fitness = [prec1, size]
        print("FITNESS:", fitness)
        return fitness

    @staticmethod
    def value(fitness_vector, objective_index):
        """
        This is a static method required by NSGA-II for sorting populations
        based on a given fitness function, or for returning a given index of a
        population based on a given fitness function.

        :param fitness_vector: A vector/list of fitnesses.
        :param objective_index: The index of the desired fitness.
        :return: The fitness at the objective index of the fitness vecror.
        """

        if not isinstance(fitness_vector, list):
            return float("inf")

        return fitness_vector[objective_index]

class face(Dataset):
    def __init__(self, hdf5_file):
        hdf5_list = hdf5_file  # only h5 files
        print('h5 list ', hdf5_list)
        self.datasets = []
        self.datasets_gt = []
        self.total_count = 0
        self.limits = []
        for f in hdf5_list:
            h5_file = h5py.File(f, 'r')
            dataset = h5_file['data']
            dataset_gt = h5_file['label']
            self.datasets.append(dataset)
            self.datasets_gt.append(dataset_gt)
            self.limits.append(self.total_count)
            self.total_count += len(dataset)
            #print ('len ',self.datasets.shape)
            #print (self.limits )

    def __getitem__(self, index):

        dataset_index = -1
        #print ('index ',index)
        for i in range(len(self.limits) - 1, -1, -1):
            if index >= self.limits[i]:
                dataset_index = i
                break
        #print ('dataset_index ',dataset_index)
        assert dataset_index >= 0, 'negative chunk'

        in_dataset_index = index - self.limits[dataset_index]
        #print('in_dataset_index ', in_dataset_index )
        return self.datasets[dataset_index][in_dataset_index], self.datasets_gt[dataset_index][in_dataset_index]

    def __len__(self):
        return self.total_count


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


def save_checkpoint(state, filename):
    torch.save(state, filename)


def train(train_loader, model, criterion, optimizer, epoch, settings):
    correct = 0
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.long()
        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output, name = model(input_var.float(), settings)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).cpu().sum()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))

    print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(correct,
                                                 len(train_loader.dataset), 100. * correct / len(train_loader.dataset)))


def test(test_loader, criterion, model, settings):
    correct = 0
    test_loss = 0
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):

        target = target.long()

        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output, name = model(input_var.float(), settings)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).cpu().sum()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(i, len(test_loader), batch_time=batch_time, loss=losses))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        losses.avg, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    res = correct / len(test_loader.dataset)
    return res, losses.avg, name

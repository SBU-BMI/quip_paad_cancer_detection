'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
# from utils import progress_bar
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.autograd import Variable
import time
import gc
import scipy.io as sio
from random import shuffle
from model_paad import PreActResNet34, PreActResNet18, ResNet18, auc_roc
from PIL import Image
from time import strftime
import collections
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc, f1_score
import random

parser = argparse.ArgumentParser(description='noisy CIFAR-10 training:')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of images in each mini-batch')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate for SGD')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay parameter')
parser.add_argument('--epoch', type=int, default=20,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--out', default='result_paad_baseline',
                    help='Directory to output the result')
parser.add_argument('--begin', type=int, default=30,  # dont update baseline
                    help='When to begin updating labels')
parser.add_argument('--stop', type=int, default=200,
                    help='When to stop updating labels')
parser.add_argument('--asym', action='store_true',
                    help='Asymmetric noise')
parser.add_argument('--seed', type=int, default=123987,
                    help='Random Seed')
parser.add_argument('--note', default='',
                    help='note to job')
parser.add_argument('--gpu', type=int, default=5,
                    help='gpu id')
args = parser.parse_args()
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

with open(os.path.basename(__file__)) as f:
    codes = f.readlines()
print('\n\n' + '='*20 + os.path.basename(__file__) + '='*20)
codes = [c[:-1] for c in codes]
for c in codes:
    print(c)
print(str(args))

os.environ['CUDA_VISIBLE_DEVICES']='{}'.format(args.gpu)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_auc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_class = 2

mean = [0.7238, 0.5716, 0.6779]
std = [0.1120, 0.1459, 0.1089]
input_size = 224
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(22),
        transforms.CenterCrop(350),
        transforms.Scale(input_size),
        transforms.RandomHorizontalFlip(),  # simple data augmentation
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),

    'val': transforms.Compose([
        transforms.CenterCrop(350),
        transforms.Scale(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

class val_loader(Dataset):
    def __init__(self, base, transform=None):
        self.base = base
        self.transform = transform

    def __getitem__(self, i):
        image, label = self.base[i]
        image = Image.open(image).convert('RGB')     # ori: RGB, do not convert to numpy, keep it as PIL image to apply transform

        if self.transform:
            image = self.transform(image)
            if image.size()[0] == 1:     # this is grayscale image, add 2 more channels
                image = torch.cat((image, image, image), dim = 0)

        return image, label

    def __len__(self):
        return len(self.base)


class train_loader(Dataset):
    def __init__(self, base, args, transform=None):
        self.base = base
        self.args = args
        self.count = 0
        self.pred_avg = 10
        self.labels = np.zeros(len(self.base), dtype=np.int32)
        self.soft_labels = np.zeros((len(self.base), num_class), dtype=np.float32)
        self.prediction = np.zeros((len(self.base), self.pred_avg, num_class), dtype=np.float32)
        self.transform = transform

        for idx in range(len(self.base)):
            _, label = self.base[idx]
            self.labels[idx] = label
            self.soft_labels[idx][self.labels[idx]] = 1.

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        image, _ = self.base[i]     # image is PIL object
        image = Image.open(image).convert('RGB')

        if self.transform:
            image = self.transform(image)
            if image.size()[0] == 1:     # this is grayscale image, add 2 more channels
                image = torch.cat((image, image, image), dim = 0)

        return image, self.labels[i], self.soft_labels[i], i

    def label_update(self, results):
        self.count += 1
        # While updating the noisy label y_i by the probability s,
        # we used the average output probability of the network of the past 10 epochs as s.
        idx = (self.count - 1) % self.pred_avg
        self.prediction[:, idx] = results

        if self.args.begin <= self.count < self.args.stop:
            self.soft_labels = self.prediction.mean(axis=1)
            self.labels = np.argmax(self.soft_labels, axis=1).astype(np.int32)

        if self.count == self.args.epoch:
            if not (os.path.isdir(self.args.out)): os.system('mkdir ' + self.args.out)
            np.save('{}/labels_last.npy'.format(self.args.out), self.labels)
            np.save('{}/soft_labels_last.npy'.format(self.args.out), self.soft_labels)


def load_fns(file_list, parent):
    fns = [f.strip() for f in open(file_list, 'r')]
    fns = list(set(fns))        # remove duplicate
    fns = [f.split() for f in fns]
    res = [[os.path.join(parent, fn[0]), int(float(fn[1]) > 0.5)] for fn in fns if len(fn) > 1]
    return res


model_save = 'paad_baseline_preact-res34_train_TCGA'

train_parent = 'tumor_data_2400_40X/tcga_ensemble'
val_parent = 'tumor_data_2400_40X/test_TCGA_patches'

test_list = load_fns('label_TCGA.txt', val_parent)

# train_list = load_fns(os.path.join(train_parent, 'label.txt'), train_parent)
# random.shuffle(train_list)
# train_list = train_list[:int(len(train_list)/2)]
# with open(os.path.join(train_parent, 'train_list.txt'), 'w') as f:
#     for fn, lb in train_list:
#         f.writelines('{} {}\n'.format(fn, lb))

train_list = load_fns(os.path.join(train_parent, 'train_list.txt'), '.')

train_parent_seer = 'tumor_data_2400_40X/tumor_2400_40X_selected_1'
train_list_seer = load_fns('train_list_2pos_5neg_exclusive.txt', train_parent_seer)
train_list += train_list_seer

testset = val_loader(test_list, transform=data_transforms['val'])
trainset = train_loader(train_list, args=args, transform=data_transforms['train'])
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=False)


test_seer = load_fns('test_single_mixed.txt', 'tumor_data_2400_40X/tumor_2400_40X_selected_1')

print('\n\nLen of test: {}\nLen of test_seer: {}\ntrain: {}\n'.format(len(test_list), len(test_seer), len(train_list)))
print('test /  train:')
for data_list in [test_list, test_seer, train_list]:
    C = collections.defaultdict(int)
    for _, lb in data_list:
        C[lb] += 1
    print(C)

testset_seer = val_loader(test_seer, transform=data_transforms['val'])
test_loader_seer = DataLoader(testset_seer, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=False)


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def build_model(num_class=num_class):
    net = PreActResNet34(num_class - 1)
    if torch.cuda.is_available():
        net.cuda()
        torch.backends.cudnn.benchmark = True
    opt = torch.optim.SGD(net.params(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    return net, opt


net, opt = build_model()


def train(epoch):
    train_loss = 0
    correct = 0
    total = 0
    start = time.time()
    net.train()

    if epoch < 4:
        lr = args.lr
    elif epoch < 8:
        lr = args.lr / 2
    elif epoch < 10:
        lr = args.lr / 10
    elif epoch < 15:
        lr = args.lr / 50
    else:
        lr = args.lr / 100

    opt = optim.SGD(filter(lambda p: p.requires_grad, net.params()), lr=lr, momentum=0.9, weight_decay=args.weight_decay)

    for batch_idx, (image, labels, soft_label, ind) in enumerate(train_loader):
        image = to_var(image, requires_grad=False)
        labels = to_var(labels.type(torch.FloatTensor), requires_grad=False)

        outputs = net(image)
        loss = F.binary_cross_entropy_with_logits(outputs, labels, reduce=False)
        loss = torch.mean(loss)

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss += loss.item()
        predicted = (F.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 100 == 0:
            print('\t| Epoch: {} \tBatch_idx: [{}/{}]\tLr: {}\tLoss: {:.4f}\tAccuracy: {:.4f}\tTime: {:.2f} mins'.format(
                epoch, batch_idx, len(train_loader), lr, train_loss / (batch_idx + 1), 100. * correct / total,
                                         (time.time() - start) / 60.0))

    print('| Train \tEpoch: {}\tLr: {}\tLoss: {:.4f}\tAccuracy: {:.4f}\tTime: {:.2f} mins'.format(
            epoch, lr, train_loss / (batch_idx+1), 100.*correct/total, (time.time() - start) / 60.0))


def test(epoch, testloader):
    global best_auc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    start = time.time()
    for param_group in opt.param_groups:
        lr = param_group['lr']
        break

    nline = 0
    Pr = np.empty(shape = (20000, 1), dtype = np.int32)
    Or = np.empty(shape = (20000, 1), dtype = np.float32)
    Tr = np.empty(shape = (20000, 1), dtype = np.int32)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            targets = to_var(targets.type(torch.FloatTensor), requires_grad=False)
            inputs = to_var(inputs, requires_grad=False)
            outputs = net(inputs)
            loss = F.binary_cross_entropy_with_logits(outputs, targets)

            test_loss += loss.item()
            # _, predicted = outputs.max(1)
            predicted = (F.sigmoid(outputs) > 0.5).float()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            N = outputs.size(0)
            output = F.sigmoid(outputs).data.cpu().numpy()
            pred = predicted.data.cpu().numpy()

            Pr[nline:nline + N] = pred.reshape(-1, 1)
            Or[nline:nline + N] = output.reshape(-1, 1)
            Tr[nline:nline + N] = targets.data.cpu().numpy().reshape(-1, 1)
            nline += N

    Pr = Pr[:nline]
    Or = Or[:nline]
    Tr = Tr[:nline]
    # val_acc = accuracy_score(Tr, Pr)
    f1 = f1_score(Tr, Pr, average='binary')
    val_auc, fpr, tpr = auc_roc(Or, Tr)

    print('| Eval \tEpoch: {}\tLr: {}\tLoss: {:.4f}\tAccuracy: {:.4f}'
          '\tF1: {:.4f}\tAUC: {:.4f}\tTime: {:.2f} mins'.format(epoch, lr, \
            test_loss / (batch_idx + 1), correct / total, f1, val_auc, (time.time() - start) / 60.0))

    # Save checkpoint.
    state = {
        'net': net.state_dict(),
        'acc': val_auc,
        'epoch': epoch,
        'codes': codes,
    }
    if val_auc > best_auc or epoch == args.epoch - 1:
        print('Saving... Eval Epoch: {}'.format(epoch))
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + model_save + '_epoch_{}_auc_{}_best'.format(epoch, val_auc))
        best_auc = val_auc
        save_name = os.path.join(args.out, model_save + '_epoch_{}_auc_{}_best.npy'.format(epoch, val_auc))
        np.save(save_name, np.concatenate((fpr.reshape(-1, 1), tpr.reshape(-1, 1)), axis=1))
    else:
        torch.save(state, './checkpoint/' + model_save + '_epoch_{}_auc_{}'.format(epoch, val_auc))


print('Start training ...')

old_model = 'paad_lrs_preact-res34_2pos_5negs_60wsi_exclude_valSlides_hard_epoch_149_acc_77.32352941176471'
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/' + old_model)
net.load_state_dict(checkpoint['net'])
print('loaded ckpt at epoch: {}, best acc: {}'.format(checkpoint['epoch'], checkpoint['acc']))

for epoch in range(start_epoch, args.epoch):
    train(epoch)
    test(epoch, test_loader)
    test(epoch, test_loader_seer)

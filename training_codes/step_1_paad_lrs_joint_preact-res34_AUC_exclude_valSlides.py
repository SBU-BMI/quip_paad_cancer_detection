from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
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
import collections
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc, f1_score

parser = argparse.ArgumentParser(description='noisy Cancer segmentation training:')
parser.add_argument('--batch_size', type=int, default=100,
                    help='Number of images in each mini-batch')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate for SGD')
parser.add_argument('--lr_meta', type=float, default=0.001,
                    help='Learning rate for SGD of the meta learning')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay parameter')
parser.add_argument('--epoch', type=int, default=100,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--out', default='result_lrs_joint',
                    help='Directory to output the result')
parser.add_argument('--begin', type=int, default=300,  # dont update baseline
                    help='When to begin updating labels')
parser.add_argument('--stop', type=int, default=200,
                    help='When to stop updating labels')
parser.add_argument('--asym', action='store_true',
                    help='Asymmetric noise')
parser.add_argument('--seed', type=int, default=123987,
                    help='Random Seed')
parser.add_argument('--note', default='',
                    help='note to job')
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
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

        return image, int(label)

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
            self.labels[idx] = int(label)
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
    fns = [f.strip().split() for f in open(file_list, 'r')]
    res = [[os.path.join(parent, fn[0]), fn[1]] for fn in fns]
    return res


model_save = 'paad_lrs_preact-res34_60wsi_exclude_valSlides_5pos_5negs_balanced_hard'

train_parent = 'tumor_data_2400_40X/tumor_2400_40X_selected_1'
# val_parent = 'tumor_data_2400_40X/val_test_data'

val_clean = load_fns('clean_val_list_hard.txt', train_parent)
test_list = load_fns('test_single_hard.txt', train_parent)
train_list = load_fns('train_list_5pos_5negs_exclusive_balanced.txt', train_parent)

# train_list = load_fns('test_list_new.txt', train_parent)

print('Len of val_clean: {}\n test: {}\n train: {}\n'.format(len(val_clean), len(test_list), len(train_list)))
print('val_clean / test /  train:')
for data_list in [val_clean, test_list, train_list]:
    C = collections.defaultdict(int)
    for _, lb in data_list:
        C[lb] += 1
    print(C)

testset = val_loader(test_list, transform=data_transforms['val'])
trainset = train_loader(train_list, args=args, transform=data_transforms['train'])
val_clean_set = val_loader(val_clean, transform=data_transforms['train'])

train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
val_clean_loader = DataLoader(val_clean_set, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)


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
    results = np.zeros((len(trainset), num_class), dtype=np.float32)

    for param_group in opt.param_groups:
        lr = param_group['lr']
        break

    val_clean_loader_iter = iter(val_clean_loader)
    for batch_idx, (image, labels, soft_label, ind) in enumerate(train_loader):
        ind = list(ind.cpu().data.numpy())
        net.train()

        meta_net = PreActResNet34(num_class - 1)
        meta_net.load_state_dict(net.state_dict())

        if torch.cuda.is_available():
            meta_net.cuda()

        image = to_var(image, requires_grad=False)
        labels = to_var(labels.type(torch.FloatTensor), requires_grad=False)
        # soft_label = to_var(soft_label.type(torch.FloatTensor), requires_grad=False)

        # val_data, val_labels = next(val_clean_loader_iter)

        try:
            val_data, val_labels = next(val_clean_loader_iter)
        except StopIteration:
            val_clean_loader_iter = iter(val_clean_loader)
            val_data, val_labels = next(val_clean_loader_iter)

        val_data = to_var(val_data, requires_grad=False)
        val_labels = to_var(val_labels.type(torch.FloatTensor), requires_grad=False)

        # Lines 4 - 5 initial forward pass to compute the initial weighted loss
        y_f_hat = meta_net(image)
        cost = F.binary_cross_entropy_with_logits(y_f_hat, labels, reduce=False)
        # cost = criterion_notReduced(y_f_hat, labels)

        eps = to_var(torch.zeros(cost.size()))
        l_f_meta = torch.sum(cost * eps)

        meta_net.zero_grad()

        # Line 6 perform a parameter update
        grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
        meta_net.update_params(args.lr_meta, source_params=grads)

        # Line 8 - 10 2nd forward pass and getting the gradients with respect to epsilon
        y_g_hat = meta_net(val_data)
        l_g_meta = F.binary_cross_entropy_with_logits(y_g_hat, val_labels)
        # l_g_meta = criterion_Reduced(y_g_hat, val_labels)
        grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]

        # Line 11 computing and normalizing the weights
        w_tilde = torch.clamp(-grad_eps, min=0)
        norm_c = torch.sum(w_tilde)

        if norm_c != 0:
            w = w_tilde / norm_c
        else:
            w = w_tilde

        # Lines 12 - 14 computing for the loss with the computed weights
        # and then perform a parameter update
        outputs = net(image)
        loss = F.binary_cross_entropy_with_logits(outputs, labels, reduce=False)
        # loss = criterion_notReduced(y_f_hat, labels)
        loss = torch.sum(loss*w)

        s = torch.sigmoid(outputs)

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss += loss.item()
        # _, predicted = outputs.max(1)

        predicted = (F.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        s = s.cpu().data.numpy().reshape(-1, 1)
        results[ind] = np.concatenate((1 - s, s), axis=1)

        meta_net = None; val_data = None; val_labels = None; gc.collect()

        if batch_idx % 100 == 0:
            print('|\tEpoch: {} \tBatch_idx: [{}/{}]\tLr: {}\tLr_meta: {}\tLoss: {:.4f}\tAccuracy: {:.4f}\tTime: {:.2f} mins'.format(
                epoch, batch_idx, len(train_loader), lr, args.lr_meta, train_loss / (batch_idx + 1), 100. * correct / total,
                                         (time.time() - start) / 60.0))
            sys.stdout.flush()

    trainset.label_update(results)
    print('| Train \tEpoch: {}\tLr: {}\tLr_meta: {}\tLoss: {:.4f}\tAccuracy: {:.4f}\tTime: {:.2f} mins'.format(
            epoch, lr, args.lr_meta, train_loss / (batch_idx+1), 100.*correct/total, (time.time() - start) / 60.0))


def test(epoch, testloader):
    global best_acc
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

    print('| Eval \tEpoch: {}\tLr: {}\tLr_meta: {}\tLoss: {:.4f}\tAccuracy: {:.4f}'
          '\tF1: {:.4f}\tAUC: {:.4f}\tTime: {:.2f} mins'.format(epoch, lr, args.lr_meta, \
            test_loss / (batch_idx + 1), correct / total, f1, val_auc, (time.time() - start) / 60.0))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc or epoch == args.epoch - 1:
        print('Saving... Eval Epoch: {}'.format(epoch))
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'codes': codes,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + model_save + '_epoch_{}_acc_{}'.format(epoch, acc))
        best_acc = acc

        if not (os.path.isdir(args.out)): os.system('mkdir ' + args.out)
        # np.save('{}/labels_best.npy'.format(args.out), trainset.labels)
        # np.save('{}/soft_labels_best.npy'.format(args.out), trainset.soft_labels)

        save_name = os.path.join(args.out, model_save + '_epoch_{}_acc_{}.npy'.format(epoch, acc))
        np.save(save_name, np.concatenate((fpr.reshape(-1, 1), tpr.reshape(-1, 1)), axis=1))


print('Start training ...')
for epoch in range(start_epoch, start_epoch + args.epoch):
    train(epoch)
    test(epoch, test_loader)

for (ratio, max_epoch) in [(10, 25), (100, 25)]:
    # print('==> Resuming from checkpoint..')
    # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    # checkpoint = torch.load('./checkpoint/' + model_save)
    # net.load_state_dict(checkpoint['net'])
    # best_acc = checkpoint['acc']
    # load_epoch = checkpoint['epoch']
    # print('loaded ckpt at epoch: ', load_epoch)

    start_epoch = args.epoch
    args.epoch += max_epoch

    args.lr_meta = args.lr_meta/ratio
    opt = optim.SGD(net.params(), lr=args.lr/ratio, momentum=0.9, weight_decay=args.weight_decay)
    for param_group in opt.param_groups:
        lr = param_group['lr']; break

    print('lr: {} \t lr_meta: {}'.format(lr, args.lr_meta))
    for epoch in range(start_epoch, args.epoch):
        train(epoch)
        test(epoch, test_loader)

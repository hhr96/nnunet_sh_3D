import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from glob import glob

import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs
import losses
from dataset import Dataset
from metrics import iou_score, recall, adapted_rand_index, variation_of_info, betti_number
from utils import AverageMeter, str2bool

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


# model saving criteria is Jaccard Index = dice/(2-dice)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=3, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False,
                        type=str2bool)  # unet++有四层子网络，这个就是控制两种运行模式，一种是只看最外层的loss，一种是四层loss平均，但iou的计算都是只看最外层，所以对于模型的保存条件并没有不同
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=512, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=512, type=int,
                        help='image height')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='ICA',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-6, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,  # 沿着之前斜率先落一段，稍微快一点的sgd，但是如果很窄的minimum容易错过，因为先跳了之前的斜率
                        help='momentum')
    parser.add_argument('--nesterov', default=False, type=str2bool,  # 计算未来超前点的斜率，更快的sgd，过minimum会跳回来
                        help='nesterov')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        # 当越接近最低值得时候，weight decay可以让更新的值变小， 更不容易出minimum https://towardsdatascience.com/why-adamw-matters-736223f31b5d, L2cost
                        help='weight decay')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='adam beta1')
    parser.add_argument('--beta2', default=0.999, type=float,
                        help='adam beta2')

    # scheduler
    parser.add_argument('--scheduler', default='ConstantLR',  # 四种lr的调整方式
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-8, type=float,
                        help='minimum learning rate')  # lr按照cosine曲线趋势减少，减到min_lr为止
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2,
                        type=int)  # Reduce learning rate when a metric has stopped improving. 这里就是两个epochs后模型无法提高，lr就乘上一个值降低
    parser.add_argument('--milestones', default='1,2', type=str)  # lr按照milestones定的epoch时减小
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=2, type=int)
    # added
    parser.add_argument('--sample_freq', default=5, type=int,
                        help='sample during validation')
    parser.add_argument('--val_set_ratio', default=0.1, type=float,
                        help='portion from training set to be validation set')
    parser.add_argument('--alpha', default=0.35, type=float,
                        help='Energy loss function')
    parser.add_argument('--sigma', default=0.1, type=float,
                        help='Energy loss function')

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _, _ in train_loader:
        input = input.cuda()
        target = target.cuda()  # 读图 进dataset.py 把图放到gpu里

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            if config['loss'] == 'EnergyLoss' or config['loss'] == 'EnergyLoss_new':
                print(output.shape)
                print(output.max(), output.min())
                output_temp = torch.sigmoid(output)
                print(output_temp.max(), output_temp.min())
                score1 = output_temp[:, 0, :, :]  # prob for class target 分布应该是0到1
                score2 = (0.5 - score1)
                loss = criterion(score2, target)
                iou = iou_score(output[:, 0, :, :], target)
            else:
                loss = criterion(output, target)
                iou = iou_score(output, target)  # 跑网络 得到loss

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()  # loss回传
        optimizer.step()  # 模型参数更新

        avg_meters['loss'].update(loss.item(), input.size(0))  # 这里input.size(0)其实就是图像的通道数量
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),  # 本质上还是对所有batch_size的损失取平均，batch_size会在计算中被消除，并没有啥用
        ])
        pbar.set_postfix(postfix)  # tqdm进度条的设定
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion, epoch, writer, num):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'ou': AverageMeter(),
                  'betti_num': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                if config['loss'] == 'EnergyLoss' or config['loss'] == 'EnergyLoss_new':
                    print(output.shape)
                    print(output.max(), output.min())
                    output_temp = torch.sigmoid(output)
                    print(output_temp.max(), output_temp.min())
                    score1 = output_temp[:, 0, :, :]  # prob for class target 分布应该是0到1
                    score2 = (0.5 - score1)
                    loss = criterion(score2, target)
                    iou = iou_score(torch.unsqueeze(output[:, 0, :, :], 1), target)
                    rec = recall(torch.unsqueeze(output[:, 0, :, :], 1), target)
                    over, under = variation_of_info(torch.unsqueeze(output[:, 0, :, :], 1), target)
                    ou = over + under
                    R_I, adapted_p, adapted_r = adapted_rand_index(torch.unsqueeze(output[:, 0, :, :], 1), target)
                    betti_num = betti_number(torch.unsqueeze(output[:, 0, :, :], 1), target)
                else:
                    loss = criterion(output, target)
                    iou = iou_score(output, target)  # 跑网络 得到loss
                    rec = recall(output, target)
                    over, under = variation_of_info(output, target)
                    ou = over+under
                    R_I, adapted_p, adapted_r = adapted_rand_index(output, target)
                    betti_num = betti_number(output, target)

            writer.add_scalar('val/iou', iou, epoch*num)
            writer.add_scalar('val/recall', rec, epoch*num)
            writer.add_scalar('val/over', over, epoch*num)
            writer.add_scalar('val/under', under, epoch*num)
            writer.add_scalar('val/R_I', R_I, epoch*num)
            writer.add_scalar('val/adapted_p', adapted_p, epoch*num)
            writer.add_scalar('val/adapted_r', adapted_r, epoch*num)
            writer.add_scalar('val/betti', betti_num, epoch*num)


            # 这里没有optimization的步骤
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['ou'].update(ou, input.size(0))
            avg_meters['betti_num'].update(betti_num, input.size(0))# 更新loss跟iou的值

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('ou', avg_meters['ou'].avg),
                ('betti_num', avg_meters['betti_num'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),  # 回传
                        ('iou', avg_meters['iou'].avg),
                        ('ou', avg_meters['ou'].avg),
                        ('betti_num', avg_meters['betti_num'].avg)])


def validate_save(config, val_loader, model, criterion, epoch, val_img_ids, writer, num):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'ou': AverageMeter(),
                  'betti_num': AverageMeter()}


    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, input_gray, _ in val_loader:
            input = input.cuda()
            target = target.cuda()
            input_gray = input_gray.cuda()

            sample_image = torch.cat(
                (torch.unsqueeze(input_gray[0, :, :, :], 0), torch.unsqueeze(target[0, :, :, :], 0)), 0)

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                for layer in range(0, len(outputs)):
                    sample_image = torch.cat(
                        (sample_image, torch.unsqueeze(outputs[layer][0, :, :, :], 0)), 0)
                for i in range(1, len(val_img_ids)):
                    sample_image = torch.cat((sample_image, torch.unsqueeze(input_gray[i, :, :, :], 0),
                                              torch.unsqueeze(target[i, :, :, :], 0)), 0)
                    for layer in range(0, len(outputs)):
                        sample_image = torch.cat(
                            (sample_image, torch.unsqueeze(outputs[layer][i, :, :, :], 0)), 0)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
                sample_image = torchvision.utils.make_grid(sample_image, 6, 0)
                torchvision.utils.save_image(sample_image, 'models/%s/{}.png'.format(str(epoch)) %
                                             config['name'])
            else:

                output = model(input)
                sample_image = torch.cat(
                    (sample_image, torch.unsqueeze(output[0, 0, :, :], 0).unsqueeze(1)), 0)
                for i in range(1, target.size(0)):
                    sample_image = torch.cat((sample_image, torch.unsqueeze(input_gray[i, :, :, :], 0),
                                              torch.unsqueeze(target[i, :, :, :], 0),
                                              torch.unsqueeze(output[i, 0, :, :], 0).unsqueeze(1)), 0)
                if config['loss'] == 'EnergyLoss' or config['loss'] == 'EnergyLoss_new':
                    print(output.shape)
                    print(output.max(), output.min())
                    output_temp = torch.sigmoid(output)
                    print(output_temp.max(), output_temp.min())
                    score1 = output_temp[:, 0, :, :]  # prob for class target 分布应该是0到1
                    score2 = (0.5 - score1)
                    loss = criterion(score2, target)
                    iou = iou_score(torch.unsqueeze(output[:, 0, :, :], 1), target)
                    rec = recall(torch.unsqueeze(output[:, 0, :, :], 1), target)
                    over, under = variation_of_info(torch.unsqueeze(output[:, 0, :, :], 1), target)
                    ou = over + under
                    R_I, adapted_p, adapted_r = adapted_rand_index(torch.unsqueeze(output[:, 0, :, :], 1), target)
                    betti_num = betti_number(torch.unsqueeze(output[:, 0, :, :], 1), target)
                else:
                    loss = criterion(output, target)
                    iou = iou_score(output, target)  # 跑网络 得到loss
                    rec = recall(output, target)
                    over, under = variation_of_info(output, target)
                    ou = over + under
                    R_I, adapted_p, adapted_r = adapted_rand_index(output, target)
                    betti_num = betti_number(output, target)
                sample_image = torchvision.utils.make_grid(sample_image, 3, 0)
                torchvision.utils.save_image(sample_image, 'models/%s/{}.png'.format(str(epoch)) %
                                             config['name'])

            writer.add_scalar('val/iou', iou, epoch*num)
            writer.add_scalar('val/recall', rec, epoch*num)
            writer.add_scalar('val/over', over, epoch*num)
            writer.add_scalar('val/under', under, epoch*num)
            writer.add_scalar('val/R_I', R_I, epoch*num)
            writer.add_scalar('val/adapted_p', adapted_p, epoch*num)
            writer.add_scalar('val/adapted_r', adapted_r, epoch*num)
            writer.add_scalar('val/betti', betti_num, epoch*num)

            # 这里没有optimization的步骤
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['ou'].update(ou, input.size(0))
            avg_meters['betti_num'].update(betti_num, input.size(0))  # 更新loss跟iou的值

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('ou', avg_meters['ou'].avg),
                ('betti_num', avg_meters['betti_num'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

        return OrderedDict([('loss', avg_meters['loss'].avg),  # 回传
                            ('iou', avg_meters['iou'].avg),
                            ('ou', avg_meters['ou'].avg),
                            ('betti_num', avg_meters['betti_num'].avg)])


def main():
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)  # 判定是不是deepsupervision，设定model的name

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)  # 打印所有的config

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)  # 将config里的所有设定写入yml文件

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':  # https://blog.csdn.net/yyhhlancelot/article/details/104260794
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        if config['loss'] == 'EnergyLoss' or config['loss'] == 'EnergyLoss_new':  # https://blog.csdn.net/yyhhlancelot/article/details/104260794
            criterion = losses.__dict__[config['loss']](cuda=True, alpha=config['alpha'],sigma=config['sigma'])
        else:
            criterion = losses.__dict__[config['loss']]().cuda()  # 设定loss的方式，通过引用losses.py里的class
    # losses.py里的BCEDiceLoss就是0.5*BCEWithLogitsLoss+dice，lovazHinge就是Jaccard的loss function
    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])  # 建立model的方式，通过引用archs.py里的class

    model = model.cuda()  # Moves all model parameters and buffers to the GPU.

    params = filter(lambda p: p.requires_grad, model.parameters())  # 这里可以定义需要训练的层，此处用model.parameters()引用了全部层
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'],
                              weight_decay=config['weight_decay'])  # 设定是什么optimizer 要么是adam要么是sgd
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=True, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError  # 设置learning rate的改变方式

    # Data loading code
    train_img_ids = glob(os.path.join('inputs', config['dataset'], 'train', 'images', '*' + config['img_ext']))
    train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]  # 将图像的名字从文件里取出
    val_img_ids = glob(os.path.join('inputs', config['dataset'], 'val', 'images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(q))[0] for q in val_img_ids]  # 将图像的名字从文件里取出


    # train_img_ids, val_img_ids = train_test_split(img_ids, test_size=config['val_set_ratio'],
    #                                               random_state=41)  # 选取validation image

    train_transform = Compose([
        transforms.RandomRotate90(),
        transforms.Flip(),
        OneOf([
            transforms.HueSaturationValue(),
            transforms.RandomBrightness(),
            transforms.RandomContrast(),
        ], p=1),
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])  # augmentation设定

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'train', 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'train', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'val', 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'val', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)  # 将图像放到tensor里

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
    ])  # 训练的数据记录
    # 以上全是model运行前的设定步骤，以下才是循环的过程
    writer = SummaryWriter('models/%s/log' % config['name'])
    best_iou = 0
    best_ou = 1000
    best_betti = 1000
    trigger = 0
    for epoch in range(config['epochs']):  # epoch从零开始跳的
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)  # 就只输出两个值loss，iou
        writer.add_scalar('train/loss', train_log['loss'], epoch)
        writer.add_scalar('train/iou', train_log['iou'], epoch)
        num= len(train_loader)
        # evaluate on validation set
        if np.mod(epoch, config['sample_freq']) == 0:
            val_log = validate_save(config, val_loader, model, criterion, epoch, val_img_ids, writer, num)  # 也就只输出两个值loss，iou
        else:
            val_log = validate(config, val_loader, model, criterion, epoch, writer, num)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])  # 记录训练的log

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if epoch > 60:
            if val_log['iou'] > best_iou:  # 模型的保存条件 对比iou的最佳值
                torch.save(model.state_dict(), 'models/%s/model_iou.pth' %
                       config['name'])
                best_iou = val_log['iou']
                print("=> saved best iou model")
                trigger = 0# trigger的值在每次有best model出现就会被重置，下面的设定就是如果n次后还没有更好的model出现，就不再跑training了
            elif val_log['ou'] < best_ou:
                torch.save(model.state_dict(), 'models/%s/model_ou.pth' %
                       config['name'])
                best_ou = val_log['ou']
                print("=> saved best ou model")
                trigger = 0
            elif val_log['betti_num'] < best_betti:
                torch.save(model.state_dict(), 'models/%s/model_betti.pth' %
                       config['name'])
                best_betti = val_log['betti_num']
                print("=> saved best betti model")
                trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()
    writer.close()


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    main()

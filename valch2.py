import argparse
import os
import numpy as np
from glob import glob

import cv2
import torch
import torchvision
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score, recall, specificity, adapted_rand_index, variation_of_info, betti_number
from utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='ICA_NestedUNet_woDS',
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    config['batch_size'] = 2
    config['num_workers'] = 0

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'])
    model = model.cuda()

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'test', 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # _, val_img_ids = train_test_split(img_ids, test_size=0.9, random_state=41)
    val_img_ids = img_ids
    model.load_state_dict(torch.load('models/%s/model_ou.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'test', 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'test', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes']-1,
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()

    iou = 0
    iou_a = []
    rec_a = []
    over_a = []
    under_a = []
    betti_a = []
    rec = 0
    spec = 0
    over_temp = 0
    under_temp = 0
    over = 0
    under = 0
    betti = 0
    R_I = 0
    R_I_temp = 0
    for c in range(config['num_classes']-1):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, input_gray, meta, in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            dim = input.size(0)
            target = target.cuda()
            input_gray = input_gray.cuda()

            sample_image = torch.cat(
                (torch.unsqueeze(input_gray[0, :, :, :], 0), torch.unsqueeze(target[0, :, :, :], 0)), 0)

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                output = output[-1]
                for layer in range(0, len(outputs)):
                    sample_image = torch.cat(
                        (sample_image, torch.unsqueeze(outputs[layer][0, :, :, :], 0)), 0)
                for i in range(1, dim):
                    sample_image = torch.cat((sample_image, torch.unsqueeze(input_gray[i, :, :, :], 0),
                                              torch.unsqueeze(target[i, :, :, :], 0)), 0)
                    for layer in range(0, len(outputs)):
                        sample_image = torch.cat(
                            (sample_image, torch.unsqueeze(outputs[layer][i, :, :, :], 0)), 0)
                sample_image = torchvision.utils.make_grid(sample_image, 6, 0)
                torchvision.utils.save_image(sample_image, os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '_comb.png'))
            else:
                output = model(input)
                sample_image = torch.cat(
                    (sample_image, torch.unsqueeze(output[0, 0, :, :], 0).unsqueeze(1)), 0)
                for i in range(1, dim):
                    sample_image = torch.cat((sample_image, torch.unsqueeze(input_gray[i, :, :, :], 0),
                                              torch.unsqueeze(target[i, :, :, :], 0),
                                              torch.unsqueeze(output[i, 0, :, :], 0).unsqueeze(1)), 0)
                sample_image = torchvision.utils.make_grid(sample_image, 3, 0)
                torchvision.utils.save_image(sample_image, os.path.join('outputs', config['name'], str(c), meta['img_id'][1] + 'comb.png'))

            output = output[:, 0, :, :].unsqueeze(1)

            iou = iou_score(output, target) +iou
            iou_a.append((2*iou_score(output, target))/(iou_score(output, target)+1))
            rec_temp, rec_a_t = recall(output, target)
            rec_a.append(rec_a_t)
            spec = specificity(output, target)+spec
            over_temp, under_temp, over_a_t, under_a_t  = variation_of_info(output, target)
            over_a.append(over_a_t)
            under_a.append(under_a_t)
            R_I_temp, a_p, a_r = adapted_rand_index(output, target)
            rec = rec + rec_temp
            R_I = R_I + R_I_temp
            over = over_temp + over
            under = under_temp + under
            betti_temp, betti_a_t = betti_number(output, target)
            betti_a.append(betti_a_t)
            betti = betti+betti_temp
            avg_meter.update(iou, input.size(0))


            # output = torch.sigmoid(output).cpu().numpy()
            #
            for i in range(len(output)):
                for c in range(1):
                    # cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                    #             (output[i, c] * 255).astype('uint8'))
                    torchvision.utils.save_image(output[i, c], os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.png'))

    print('IoU: %.4f' % (iou / len(val_loader)), 'sd: %.4f'%(np.std(iou_a)))
    print('rec: %.4f' % (rec / len(val_loader)), 'sd: %.4f'%(np.std(rec_a)))
    print('over: %.4f' % (over / len(val_loader)), 'sd: %.4f'%(np.std(over_a)))
    print('under: %.4f' % (under / len(val_loader)), 'sd: %.4f'%(np.std(under_a)))
    print('betti: %.4f' % (betti / len(val_loader)), 'sd: %.4f'%(np.std(betti_a)))
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric
from skimage.metrics import adapted_rand_error, variation_of_information
from skimage.measure import label
import random


def iou_score(output, target): #Jaccard Index
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def recall(predict, target):
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict_ = predict > 0.5
    target_ = target > 0.5

    predict_ = np.atleast_1d(predict_.astype(np.bool))
    target_ = np.atleast_1d(target_.astype(np.bool))

    tp = np.count_nonzero(predict_ & target_)
    fn = np.count_nonzero(~predict_ & target_)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall


def specificity(predict, target): #Specificity，true negative rate一样
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    predict_ = predict > 0.5
    target_ = target > 0.5

    predict_ = np.atleast_1d(predict_.astype(np.bool))
    target_ = np.atleast_1d(target_.astype(np.bool))

    tn = np.count_nonzero(~predict_ & ~target_)
    fp = np.count_nonzero(predict_ & ~target_)

    try:
        specificity = tn / float(tn + fp)
    except ZeroDivisionError:
        specificity = 0.0

    return specificity

def calculate_metric_percase(pred, gt):
    if torch.is_tensor(pred):
        predict = torch.sigmoid(pred).data.cpu().numpy()
    if torch.is_tensor(gt):
        target = gt.data.cpu().numpy()

    threshold, upper, lower = 0.5, 1, 0
    predict_ = np.where(predict>threshold, upper, lower)
    target_ = target

    hd_temp = 0
    asd_temp = 0
    for i in range(len(predict)):
        predict_temp = predict_[i, 0]
        target_temp = target_[i, 0]
        a = np.zeros(predict_temp.shape)
        if np.all(predict_temp):
            hd_temp = 1000
            asd_temp = 1000
            print(predict_temp[0, 0])
            print(target_temp[0, 0])
        else:
            hd_temp = metric.binary.hd95(predict_temp, target_temp) + hd_temp
            asd_temp = metric.binary.asd(predict_temp, target_temp) + asd_temp

    hd = hd_temp/len(predict)
    asd = asd_temp/len(predict)

    # hd = metric.binary.hd95(predict_, target_)
    # asd = metric.binary.asd(predict_, target_)
    return asd, hd


def variation_of_info(pred, gt):

    if torch.is_tensor(pred):
        predict = torch.sigmoid(pred).data.cpu().numpy()
    if torch.is_tensor(gt):
        target = gt.data.cpu().numpy()

    threshold, upper, lower = 0.5, 1, 0
    predict_ = np.where(predict>threshold, upper, lower)
    target_ = target.astype(np.int32)

    splits = 0
    merges = 0
    for i in range(len(predict)):
        predict_temp = predict_[i, 0]
        target_temp = target_[i, 0]
        splits_temp, merges_temp = variation_of_information(target_temp, predict_temp)
        splits = splits + splits_temp
        merges = merges + merges_temp

    return splits/len(predict), merges/len(predict)


def adapted_rand_index(pred, gt):

    if torch.is_tensor(pred):
        predict = torch.sigmoid(pred).data.cpu().numpy()
    if torch.is_tensor(gt):
        target = gt.data.cpu().numpy()

    threshold, upper, lower = 0.5, 1, 0
    predict_ = np.where(predict>threshold, upper, lower)
    target_ = target.astype(np.int32)

    # rand_index, precision, recall = adapted_rand_error(target_, predict_)

    rand_index = 0
    precision = 0
    recall = 0
    for i in range(len(predict)):
        predict_temp = predict_[i, 0]
        target_temp = target_[i, 0]
        rand_index_temp, precision_temp, recall_temp = adapted_rand_error(target_temp, predict_temp)
        rand_index = rand_index + rand_index_temp
        precision = precision + precision_temp
        recall = recall +recall_temp

    return rand_index / len(predict), precision / len(predict), recall / len(predict)


def betti_number(pred, gt):
    if torch.is_tensor(pred):
        predict = torch.sigmoid(pred).data.cpu().numpy()
    if torch.is_tensor(gt):
        target = gt.data.cpu().numpy()

    threshold, upper, lower = 0.5, 1, 0
    predict_ = np.where(predict>threshold, upper, lower)
    target_ = target.astype(np.int32)

    pred_cc = 0
    gt_cc = 0
    betti = 0
    _, _, w, h = predict.shape

    for i in range(len(predict)):
        dim_new = random_int_list(256, 512, 15)
        for j in range(1, 10):
            w_new = h_new = dim_new[j]
            predict_temp = np.zeros((w_new, h_new), np.int32)
            predict_temp = predict_[i, 0][int((w - w_new) / 2):int(w_new + (w - w_new) / 2),
                        int((h - h_new) / 2):int(h_new + (h - h_new) / 2)]
            target_temp = np.zeros((w_new, h_new), np.int32)
            target_temp = target_[i, 0][int((w - w_new) / 2):int(w_new + (w - w_new) / 2),
                        int((h - h_new) / 2):int(h_new + (h - h_new) / 2)]
            # predict_temp = predict_[i, 0]
            # target_temp = target_[i, 0]
            labels_prd, pred_cc = label(predict_temp, return_num= True, connectivity=1)
            labels_gt, gt_cc= label(target_temp, return_num= True, connectivity=1)
            betti = betti + np.abs(pred_cc - gt_cc)

    return betti / len(predict) / 10

def random_int_list(start, stop, length):

    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0

    random_list = []

    for i in range(length):

        random_list.append(random.randint(start, stop))

    return random_list
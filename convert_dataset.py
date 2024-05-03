import os

import cv2

import paths
from dataloaders import set_fold
from utils import characterize, LABEL_GT_CHARACTERIZER


def make_fold(fold):
    train_cases, _ = set_fold(fold, 4)
    images = []
    labels = []
    for c in train_cases:
        fns = [fn for fn in os.listdir(os.path.join(paths.DATASET_DIR, c, 'label')) 
                if fn.endswith('.png') or fn.endswith('.jpg')]
        fns.sort(key=lambda x: int(x[:-4]))
        images += [cv2.imread(os.path.join(paths.DATASET_DIR, c, fn)) for fn in fns]
        labels += [cv2.imread(os.path.join(paths.DATASET_DIR, c, 'label', fn)) for fn in fns]
    return images, labels


def convert_to_pns():
    target_dir = '../PNS-Net-main/dataset/'
    for fold in range(4):
        fold_kidney_dir = os.path.join(target_dir, 'fold{}_kidney'.format(fold))
        fold_tumor_dir = os.path.join(target_dir, 'fold{}_tumor'.format(fold))
        os.makedirs(os.path.join(fold_kidney_dir, 'Frame'), exist_ok=True)
        os.makedirs(os.path.join(fold_kidney_dir, 'GT'), exist_ok=True)
        os.makedirs(os.path.join(fold_tumor_dir, 'Frame'), exist_ok=True)
        os.makedirs(os.path.join(fold_tumor_dir, 'GT'), exist_ok=True)
        images, labels = make_fold(fold)
        for i, img in enumerate(images):
            cv2.imwrite(os.path.join(fold_kidney_dir, 'Frame', '{}.png'.format(i)), img)
            cv2.imwrite(os.path.join(fold_tumor_dir, 'Frame', '{}.png'.format(i)), img)
        for i, lbl in enumerate(labels):
            ch_label = characterize(lbl.transpose((2, 0, 1)), LABEL_GT_CHARACTERIZER)
            cv2.imwrite(os.path.join(fold_kidney_dir, 'GT', '{}.png'.format(i)), ch_label[0] * 255)
            cv2.imwrite(os.path.join(fold_tumor_dir, 'GT', '{}.png'.format(i)), ch_label[1] * 255)
    
    
if __name__ == '__main__':
    convert_to_pns()

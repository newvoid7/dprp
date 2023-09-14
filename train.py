import os

import numpy as np
import torch
from torch.nn import MSELoss
import torch.nn.functional as nnf
from torchvision import transforms
import cv2

from network.profen import ProFEN
from network.affine2d import Affine2dPredictor, Affine2dTransformer
from network.loss import MultiCELoss, RefInfoNCELoss, InfoNCELoss
from network.vos import LSTMVOSWithSC
from dataloaders import VOSDataloader, ProbeSingleCaseDataloader
from probe import deserialize_probes, Probe, ablation_num_of_probes
from utils import resized_center_square, cv2_to_tensor, select_best_weight
import paths

                
def set_fold(fold, num_all_folds):
    num_all_cases = len(paths.ALL_CASES)
    fold_size = num_all_cases // num_all_folds
    test_indices = [fold * fold_size + i for i in range(fold_size)]
    test_cases = [paths.ALL_CASES[i] for i in test_indices]
    train_indices = [i for i in range(len(paths.ALL_CASES)) if i not in test_indices]
    train_cases = [paths.ALL_CASES[i] for i in train_indices]
    return train_cases, test_cases


class AgentTask:
    def __init__(self, mask_path):
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        self.mask = resized_center_square(self.mask, out_size=512)
        self.mask = cv2_to_tensor(self.mask)

    def apply(self, i):
        """
        Actually the agent task should be rendering from a near position, but we use 2d data augmentation instead.
        Because profen reduces 2 DoF, the agent task should reduce the remaining 4 DoF (3 translation and 1 rotation).
        Also, in endoscopic images, some mask (due to the viewport or something else) exist, so we use a mask to help.
        Args:
            i (torch.Tensor): shape of (B, C, H, W)
        Returns:
            torch.Tensor:
        """
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=40, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomResizedCrop(size=512, scale=(0.4, 1.1), ratio=(1.0, 1.0),
                                         interpolation=transforms.InterpolationMode.NEAREST)
        ])
        i = transform(i)
        if self.mask.size() != i.size()[-2:]:
            mask = nnf.interpolate(self.mask, size=i.size()[-2:])
        else:
            mask = self.mask.to(i.device)
        i = i * mask
        return i


class VOSAugmentor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=30, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomResizedCrop(size=512, scale=(0.8, 1.1), ratio=(1.0, 1.0),
                                         interpolation=transforms.InterpolationMode.NEAREST)
        ])
        return

    def __call__(self, vos_batch):
        images = vos_batch['data']
        labels = vos_batch['seg']
        bs, seq_len, ch, height, width = images.shape
        assemble = torch.cat([images.reshape((-1, height, width)),
                              labels.reshape((-1, height, width))], dim=0)
        augmented = self.transform(assemble)
        augmented_images = augmented[:len(augmented) // 2]
        augmented_labels = augmented[len(augmented) // 2:]
        augmented_images = augmented_images.reshape((bs, seq_len, ch, height, width))
        augmented_labels = augmented_labels.reshape((bs, seq_len, ch, height, width))
        return augmented_images, augmented_labels


def train_profen(base_dir=paths.DATASET_DIR, fold=0, n_folds=6, n_epoch=300, batch_size=8,
                 use_ref_info_nce=True):
    """
    Use the positions of probes as reference to decide how bad the negative pair is (see network.loss.RefInfoNCELoss).
    Actually, the reference should be all render parameters of the probe, including rotation quaternion.
    But in our task, the probes all focus at (0,0,0), the computation of the quaternion similarity is not necessary.
    Note that the agent task should be same as train_affine2d.
    """
    fe = ProFEN().cuda()
    fe.train()
    loss_func = RefInfoNCELoss() if use_ref_info_nce else InfoNCELoss()
    optimizer = torch.optim.Adam(fe.parameters(), lr=1e-4)
    train_cases, _ = set_fold(fold, n_folds)
    os.makedirs('weights/fold{}'.format(fold), exist_ok=True)
    # dataloader = ProbeSingleCaseDataloader(
    #     [ablation_num_of_probes(
    #         deserialize_probes(os.path.join('results', case_id, 'probes.pk')), factor=4)
    #         for case_id in train_cases],
    #     batch_size=batch_size)
    dataloader = ProbeSingleCaseDataloader(
        [deserialize_probes(os.path.join(paths.RESULTS_DIR, case_id, paths.PROBE_FILENAME))
        for case_id in train_cases],
        batch_size=batch_size)
    agent = AgentTask(mask_path=os.path.join(base_dir, 'mask.png'))
    epoch_loss = []
    for epoch in range(n_epoch):
        iter_loss = []
        for i in range(dataloader.num_total // batch_size):
            optimizer.zero_grad()
            batch = next(dataloader)
            render = torch.from_numpy(batch['data']).float().cuda()
            noise = agent.apply(render)
            positions = torch.from_numpy(batch['position']).float().cuda()
            features = fe(torch.cat([render, noise], dim=0))
            if use_ref_info_nce:
                loss = loss_func(features[:len(features) // 2], features[len(features) // 2:], positions)
            else:
                loss = loss_func(features[:len(features) // 2], features[len(features) // 2:])
            iter_loss.append(float(loss.data))
            loss.backward()
            optimizer.step()
        iter_loss = np.asarray(iter_loss)
        epoch_loss.append(iter_loss)
        print('epoch[{}/{}], loss: {}'.format(epoch + 1, n_epoch, iter_loss.mean()))
        if (epoch + 1) % 10 == 0:
            torch.save(fe.state_dict(), '{}/fold{}/profen{}_{}.pth'
                       .format(paths.WEIGHTS_DIR, fold, '' if use_ref_info_nce else '_infonce', epoch + 1))
    np.save('{}/fold{}/profen{}.npy'.format(paths.WEIGHTS_DIR, fold, '' if use_ref_info_nce else '_infonce'),
            np.asarray(epoch_loss))
    return


def train_affine2d(base_dir=paths.DATASET_DIR, fold=0, n_folds=6, n_epoch=300, batch_size=8):
    predictor = Affine2dPredictor().cuda()
    predictor.train()
    transformer = Affine2dTransformer().cuda()
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-4)
    loss_func = MSELoss().cuda()
    train_cases, _ = set_fold(fold, n_folds)
    os.makedirs('weights/fold{}'.format(fold), exist_ok=True)
    dataloader = ProbeSingleCaseDataloader(
        [deserialize_probes(os.path.join(paths.RESULTS_DIR, case_id, paths.PROBE_FILENAME)) for case_id in train_cases],
        batch_size=batch_size)
    agent = AgentTask(mask_path=os.path.join(base_dir, 'mask.png'))
    epoch_loss = []
    for epoch in range(n_epoch):
        iter_loss = []
        for i in range(dataloader.num_total // batch_size):
            optimizer.zero_grad()
            batch = next(dataloader)
            render = torch.from_numpy(batch['data']).float().cuda()
            target = agent.apply(render)
            pred_params = predictor(render, target)
            pred_target = transformer(render, pred_params)
            # TODO: should compute the loss without mask
            # TODO: treat the channels differently
            loss = loss_func(target, pred_target)
            iter_loss.append(float(loss.data))
            loss.backward()
            optimizer.step()
        iter_loss = np.asarray(iter_loss)
        epoch_loss.append(iter_loss)
        print('epoch[{}/{}], loss: {}'.format(epoch + 1, n_epoch, iter_loss.mean()))
        if (epoch + 1) % 10 == 0:
            torch.save(predictor.state_dict(), 'weights/fold{}/affine2d_{}.pth'.format(fold, epoch + 1))
    np.save('weights/fold{}/affine2d.npy'.format(fold), np.asarray(epoch_loss))
    return


def train_vos(base_dir=paths.DATASET_DIR, fold=0, n_folds=6, n_epoch=500, batch_size=8):
    vos = LSTMVOSWithSC().cuda()
    vos.train()
    loss_func = MultiCELoss()
    optimizer = torch.optim.Adam(vos.parameters(), lr=1e-4)
    train_cases, _ = set_fold(fold, n_folds)
    dataloader = VOSDataloader(
        image_dirs=[os.path.join(base_dir, case_id) for case_id in train_cases],
        label_dirs=[os.path.join(base_dir, case_id, 'label') for case_id in train_cases],
        batch_size=batch_size
    )
    augmentor = VOSAugmentor()
    epoch_loss = []
    for epoch in range(n_epoch):
        iter_loss = []
        for i in range(dataloader.num_total // batch_size):
            optimizer.zero_grad()
            batch = next(dataloader)
            images, labels = augmentor(batch)
            images = images.cuda()
            labels = labels.cuda()
            pred = vos(images[:, 0, ...], labels[:, 0, ...], images[:, 1:, ...])
            loss = loss_func(pred, labels[:, 1:, ...])
            iter_loss.append(float(loss.data))
            loss.backward()
            optimizer.step()
        iter_loss = np.asarray(iter_loss)
        epoch_loss.append(iter_loss)
        print('epoch[{}/{}], loss: {}'.format(epoch + 1, n_epoch, iter_loss.mean()))
        if (epoch + 1) % 10 == 0:
            torch.save(vos.state_dict(), '{}/fold{}/vos_train_{}.pth'.format(paths.WEIGHTS_DIR, fold, epoch + 1))
    np.save('{}/fold{}/vos_loss.npy'.format(paths.WEIGHTS_DIR, fold), np.asarray(epoch_loss))
    return


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    for fold in range(6):
        train_profen(fold=fold, n_folds=6)
        train_profen(fold=fold, n_folds=6, use_ref_info_nce=False)
        train_affine2d(fold=fold, n_folds=6)
    #     train_vos(fold=n_fold)
    for fold in range(4):
        select_best_weight(['{}/fold{}/profen_{}.pth'.format(paths.WEIGHTS_DIR, fold, i * 10 + 10) for i in range(30)],
                           '{}/fold{}/profen.npy'.format(paths.WEIGHTS_DIR, fold),
                           '{}/fold{}/profen_best.pth'.format(paths.WEIGHTS_DIR, fold))
        select_best_weight(['{}/fold{}/profen_infonce_{}.pth'.format(paths.WEIGHTS_DIR, fold, i * 10 + 10) for i in range(30)],
                           '{}/fold{}/profen_infonce.npy'.format(paths.WEIGHTS_DIR, fold),
                           '{}/fold{}/profen_infonce_best.pth'.format(paths.WEIGHTS_DIR, fold))
        select_best_weight(['{}/fold{}/affine2d_{}.pth'.format(paths.WEIGHTS_DIR, fold, i * 10 + 10) for i in range(30)],
                           '{}/fold{}/affine2d.npy'.format(paths.WEIGHTS_DIR, fold),
                           '{}/fold{}/affine2d_best.pth'.format(paths.WEIGHTS_DIR, fold))

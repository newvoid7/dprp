import os

import numpy as np
import torch
from torch.nn import MSELoss
import torch.nn.functional as nnf
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt

from network.profen import ProFEN
from network.affine2d import Affine2dPredictor, Affine2dTransformer
from network.loss import RefInfoNCELoss, InfoNCELoss
from dataloaders import ProbeSingleCaseDataloader
from probe import deserialize_probes, Probe, ablation_num_of_probes
from utils import resized_center_square, cv2_to_tensor, time_it
import paths

                
def set_fold(fold, num_all_folds):
    num_all_cases = len(paths.ALL_CASES)
    fold_size = num_all_cases // num_all_folds
    test_indices = [fold * fold_size + i for i in range(fold_size)]
    test_cases = [paths.ALL_CASES[i] for i in test_indices]
    train_indices = [i for i in range(len(paths.ALL_CASES)) if i not in test_indices]
    train_cases = [paths.ALL_CASES[i] for i in train_indices]
    return train_cases, test_cases


class BaseTrainer:
    def __init__(self, model_name, model, save_dir, 
                 draw_loss=True, save_cycle=0, n_epoch=300, n_iter=100):
        """
        Args:
            model_name (str): 
            model (torch.nn.Module): 
            save_dir (str): Where to save the weights and training loss information.
            draw_loss (bool, optional): Draw the loss curve save to `loss.png`. Defaults to True.
            save_cycle (int, optional): Save the weights every cycle. Defaults to 0.
            n_epoch (int, optional): Defaults to 300.
            n_iter (int, optional): Defaults to 100.
        """
        self.model_name = model_name
        self.model = model
        self.save_dir = save_dir
        self.draw_loss = draw_loss
        self.save_cycle = save_cycle
        self.n_epoch = n_epoch
        self.n_iter = n_iter
        os.makedirs(save_dir, exist_ok=True)
    
    def train_iter(self) -> float:
        pass
    
    @time_it
    def train_epoch(self, epoch):
        epoch_losses = []
        for iter in range(self.n_iter):
            iter_loss = self.train_iter()
            epoch_losses.append(iter_loss)
        print('Epoch [{}/{}], average loss: {}'.format(epoch + 1, self.n_epoch, np.asarray(epoch_losses).mean()))
        if (self.save_cycle != 0) and ((epoch + 1) % self.save_cycle == 0):
            torch.save(self.model.state_dict(), '{}/{}.pth'.format(self.save_dir, epoch))
        torch.save(self.model.state_dict(), '{}/last.pth'.format(self.save_dir))
        return np.asarray(epoch_losses)
    
    @time_it
    def train(self, resume=False):
        """ Call train_epoch
        Args:
            resume (bool, optional): Resume from the last epoch. Defaults to False.
        """
        print('===> Training {}'.format(self.model_name))
        print('Save directory is {}.'.format(self.save_dir))
        self.model.cuda()
        self.model.train()
        if resume:
            train_losses = np.load('{}/loss.npy'.format(self.save_dir))
            last_epoch = len(train_losses)
            self.model.load_state_dict(torch.load('{}/last.pth'.format(self.save_dir)))
            print('Resumed from epoch {}.'.format(last_epoch))
        else:
            last_epoch = 0
            train_losses = np.zeros((0, self.n_iter))
        best_loss = np.inf
        for epoch in range(last_epoch, self.n_epoch):
            epoch_losses = self.train_epoch(epoch)
            if best_loss > epoch_losses.mean():
                torch.save(self.model.state_dict(), '{}/best.pth'.format(self.save_dir))
            train_losses = np.append(train_losses, [epoch_losses], axis=0)
            np.save('{}/loss.npy'.format(self.save_dir), train_losses)
        if self.draw_loss:
            plt.figure()
            plt.plot(np.arange(len(train_losses)), train_losses.mean(-1))
            plt.title('{} Loss Curve'.format(self.model_name))
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig('{}/loss.png'.format(self.save_dir))
        print('===> Training {} done.'.format(self.model_name))
        return


class ProfenTrainer(BaseTrainer):
    def __init__(self, ablation=None, fold=0, n_folds=0, batch_size=8):
        """
        Args:
            ablation (str, optional):
                'wo_ref_loss': use original InfoNCE loss rather than RefInfoNCE
                'div_4': use 1/4 of the probes
                'div_9': use 1/9 of the probes
                'div_16': use 1/16 of the probes
        """
        model = ProFEN()
        model_name = 'profen' if ablation is None else 'profen_' + ablation
        save_dir = os.path.join(paths.WEIGHTS_DIR, 'fold{}'.format(fold), model_name)
        self.with_ref_loss = ablation != 'wo_ref_loss'
        self.loss_func = RefInfoNCELoss().cuda() if self.with_ref_loss else InfoNCELoss().cuda()
        train_cases, _ = set_fold(fold, n_folds)
        probes = [deserialize_probes(os.path.join(paths.RESULTS_DIR, case_id, paths.PROBE_FILENAME))
            for case_id in train_cases]
        if ablation == 'div_4':
            probes = [ablation_num_of_probes(p_list, 2) for p_list in probes]
        elif ablation == 'div_9':
            probes = [ablation_num_of_probes(p_list, 3) for p_list in probes]
        elif ablation == 'div_16':
            probes = [ablation_num_of_probes(p_list, 4) for p_list in probes]
        self.dataloader = ProbeSingleCaseDataloader(probes=probes, batch_size=batch_size)
        self.batch_size = batch_size
        n_iter = self.dataloader.num_total // batch_size
        self.agent = AgentTask(mask_path=os.path.join(paths.DATASET_DIR, 'mask.png'))
        super().__init__(model_name=model_name, model=model, save_dir=save_dir, n_iter=n_iter)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
                
    def train_iter(self) -> float:
        self.optimizer.zero_grad()
        batch = next(self.dataloader)
        render = torch.from_numpy(batch['data']).float().cuda()
        noise = self.agent.apply(render)
        positions = torch.from_numpy(batch['position']).float().cuda()
        features = self.model(torch.cat([render, noise], dim=0))
        if self.with_ref_loss:
            loss = self.loss_func(features[:len(features) // 2], features[len(features) // 2:], positions)
        else:
            loss = self.loss_func(features[:len(features) // 2], features[len(features) // 2:])
        loss.backward()
        self.optimizer.step()
        return float(loss)


class Affine2DTrainer(BaseTrainer):
    def __init__(self, ablation=None, fold=0, n_folds=0, batch_size=8):
        model = Affine2dPredictor()
        model_name = 'affine2d' if ablation is None else 'affine2d' + ablation
        save_dir = os.path.join(paths.WEIGHTS_DIR, 'fold{}'.format(fold), model_name)
        self.transformer = Affine2dTransformer().cuda()
        self.loss_func = MSELoss()
        train_cases, _ = set_fold(fold, n_folds)
        probes = [deserialize_probes(os.path.join(paths.RESULTS_DIR, case_id, paths.PROBE_FILENAME))
            for case_id in train_cases]
        if ablation == 'div_4':
            probes = [ablation_num_of_probes(p_list, 2) for p_list in probes]
        elif ablation == 'div_9':
            probes = [ablation_num_of_probes(p_list, 3) for p_list in probes]
        elif ablation == 'div_16':
            probes = [ablation_num_of_probes(p_list, 4) for p_list in probes]
        self.dataloader = ProbeSingleCaseDataloader(probes=probes, batch_size=batch_size)
        self.batch_size = batch_size
        n_iter = self.dataloader.num_total // batch_size
        self.agent = AgentTask(mask_path=os.path.join(paths.DATASET_DIR, 'mask.png'))
        super().__init__(model_name=model_name, model=model, save_dir=save_dir, n_iter=n_iter)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
    def train_iter(self) -> float:
        self.optimizer.zero_grad()
        batch = next(self.dataloader)
        render = torch.from_numpy(batch['data']).float().cuda()
        target = self.agent.apply(render)
        pred_params = self.model(render, target)
        pred_target = self.transformer(render, pred_params)
        # TODO: should compute the loss without mask
        # TODO: treat the channels differently
        loss = self.loss_func(target, pred_target)
        loss.backward()
        self.optimizer.step()
        return float(loss)


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


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    for fold in range(0, 6):
        ProfenTrainer(fold=fold, n_folds=6).train()
        Affine2DTrainer(fold=fold, n_folds=6).train()
    for fold in range(0, 6):
        ProfenTrainer(ablation='wo_ref_loss', fold=fold, n_folds=6).train()
        ProfenTrainer(ablation='div_4', fold=fold, n_folds=6).train()
        ProfenTrainer(ablation='div_9', fold=fold, n_folds=6).train()
        ProfenTrainer(ablation='div_16', fold=fold, n_folds=6).train()
        Affine2DTrainer(ablation='div_4', fold=fold, n_folds=6).train()
        Affine2DTrainer(ablation='div_9', fold=fold, n_folds=6).train()
        Affine2DTrainer(ablation='div_16', fold=fold, n_folds=6).train()

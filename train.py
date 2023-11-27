import os
import argparse

import numpy as np
import torch
from torch.nn import MSELoss
import matplotlib.pyplot as plt

from network.profen import ProFEN
from network.transform import Affine2dPredictor, Affine2dPredictorSlim, Affine2dTransformer
from network.loss import RefInfoNCELoss, InfoNCELoss
from dataloaders import ProbeSingleCaseDataloader
from probe import ProbeGroup
from utils import time_it
from agent import AgentTask
import paths
from dataloaders import set_fold


class BaseTrainer:
    def __init__(self, model_name, model, save_dir, 
                 draw_loss=True, save_cycle=0, n_epoch=300, n_iter=None):
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
        self.n_iter = n_iter            # should be set by num_total / batch_size
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
            if len(train_losses) == 1 or len(train_losses[-1]) != len(train_losses[0]):
                train_losses = train_losses[:-1]
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
    def __init__(self, ablation=None, fold=0, n_folds=0, batch_size=8, **kwargs):
        """
        Args:
            ablation (str, optional):
                'wo_agent':
                'div_4': use 1/4 of the probes
                'div_9': use 1/9 of the probes
                'div_16': use 1/16 of the probes
        """
        model = ProFEN()
        model_name = 'profen' if ablation is None else 'profen_' + ablation
        save_dir = os.path.join(paths.WEIGHTS_DIR, 'fold{}'.format(fold), model_name)
        self.with_agent = ablation != 'wo_agent'
        self.loss_func = InfoNCELoss().cuda()
        train_cases, _ = set_fold(fold, n_folds)
        probe_groups = [ProbeGroup(deserialize_path=os.path.join(paths.RESULTS_DIR, case_id, paths.PROBE_FILENAME))
            for case_id in train_cases]
        if ablation == 'div_4':
            for pg in probe_groups:
                pg.sparse(2)
        elif ablation == 'div_9':
            for pg in probe_groups:
                pg.sparse(3)
        elif ablation == 'div_16':
            for pg in probe_groups:
                pg.sparse(4)
        probes = [pg.probes for pg in probe_groups]
        self.dataloader = ProbeSingleCaseDataloader(probes=probes, batch_size=batch_size)
        self.batch_size = batch_size
        n_iter = self.dataloader.num_total // batch_size
        self.agent = AgentTask(occlusion_dir=paths.MASK_DIR)
        super().__init__(model_name=model_name, model=model, save_dir=save_dir, n_iter=n_iter, **kwargs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
                
    def train_iter(self) -> float:
        self.optimizer.zero_grad()
        batch = next(self.dataloader)
        render = torch.from_numpy(batch['data']).float().cuda()
        noise = self.agent.apply(render) if self.with_agent else render
        features = self.model(torch.cat([render, noise], dim=0))
        loss = self.loss_func(features[:len(features) // 2], features[len(features) // 2:])
        loss.backward()
        self.optimizer.step()
        return float(loss)


class Affine2DTrainer(BaseTrainer):
    def __init__(self, ablation=None, fold=0, n_folds=0, batch_size=8, **kwargs):
        model = Affine2dPredictorSlim()
        model_name = 'affine2d' if ablation is None else 'affine2d_' + ablation
        save_dir = os.path.join(paths.WEIGHTS_DIR, 'fold{}'.format(fold), model_name)
        self.transformer = Affine2dTransformer().cuda()
        self.loss_func = MSELoss()
        train_cases, _ = set_fold(fold, n_folds)
        probe_groups = [ProbeGroup(deserialize_path=os.path.join(paths.RESULTS_DIR, case_id, paths.PROBE_FILENAME))
            for case_id in train_cases]
        if ablation == 'div_4':
            for pg in probe_groups:
                pg.sparse(2)
        elif ablation == 'div_9':
            for pg in probe_groups:
                pg.sparse(3)
        elif ablation == 'div_16':
            for pg in probe_groups:
                pg.sparse(4)
        probes = [pg.probes for pg in probe_groups]
        self.dataloader = ProbeSingleCaseDataloader(probes=probes, batch_size=batch_size)
        self.batch_size = batch_size
        n_iter = self.dataloader.num_total // batch_size
        self.agent = AgentTask(occlusion_dir=paths.MASK_DIR)
        super().__init__(model_name=model_name, model=model, save_dir=save_dir, n_iter=n_iter, **kwargs)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
    def train_iter(self) -> float:
        self.optimizer.zero_grad()
        batch = next(self.dataloader)
        render = torch.from_numpy(batch['data']).float().cuda()
        target = self.agent.apply(render)
        params = self.agent.real_params
        pred_params = self.model(render, target)
        pred_target = self.transformer(render, pred_params)
        # TODO: should compute the loss without mask
        # TODO: treat the channels differently
        loss_cycle = self.loss_func(target, pred_target)
        loss_param = self.loss_func(params, pred_params)
        loss = 0.5 * loss_cycle + 0.5 * loss_param
        loss.backward()
        self.optimizer.step()
        return float(loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--gpu', type=int, default=0, required=False, 
                        help='train on which gpu')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3], required=False, 
                        help='which folds should be trained, e.g. --folds 0 2 4')
    parser.add_argument('--n_folds', type=int, default=4, required=False, 
                        help='how many folds in total')
    parser.add_argument('--ablation', action='store_true', default=False, required=False, 
                        help='whether do the ablation')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if not args.ablation:
        for fold in args.folds:
            # ProfenTrainer(fold=fold, n_folds=args.n_folds).train()
            Affine2DTrainer(fold=fold, n_folds=args.n_folds).train()
    else:
        for fold in args.folds:
            ProfenTrainer(ablation='div_4', fold=fold, n_folds=args.n_folds).train()
            ProfenTrainer(ablation='div_9', fold=fold, n_folds=args.n_folds).train()
            ProfenTrainer(ablation='div_16', fold=fold, n_folds=args.n_folds).train()
            ProfenTrainer(ablation='wo_agent', fold=fold, n_folds=args.n_folds).train()
            # Affine2DTrainer(ablation='div_4', fold=fold, n_folds=args.n_folds).train()
            # Affine2DTrainer(ablation='div_9', fold=fold, n_folds=args.n_folds).train()
            # Affine2DTrainer(ablation='div_16', fold=fold, n_folds=args.n_folds).train()

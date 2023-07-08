import torch
import torch.nn as nn


class MultiCELoss(nn.Module):
    def __init__(self):
        super(MultiCELoss, self).__init__()
        self.smooth = torch.tensor(1e-6)

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): shape of (Batch_size, N, Channel, H, W)
            target (torch.Tensor): shape as same as pred.
        Returns:
            torch.Tensor: shape of ().
        """
        return -torch.mean(target * torch.log(pred + self.smooth))


class NCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fixed, moving):
        """
        Normalized Correlation Coefficient
        Args:
            fixed (torch.Tensor):
            moving (torch.Tensor):
        Returns:
            torch.Tensor:
        """
        f_mean = torch.mean(fixed)
        m_mean = torch.mean(moving)
        f_std = torch.std(f_mean)
        m_std = torch.std(moving)
        return ((fixed - f_mean) * (moving - m_mean) / (f_std * m_std)).sum()


class InfoNCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos_sim_func = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.eps = 1e-6
        return

    def forward(self, f, noise):
        bs = f.size()[0]
        cat_f_agent = torch.cat([noise, f], dim=0)
        # matrix: row i are 2bs pairs of f[i]: f0, f1, ..., a0, a1, ...
        matrix = torch.stack([self.cos_sim_func(f[i].repeat(bs * 2, 1), cat_f_agent)
                              for i in range(bs)], dim=0)
        matrix = torch.exp(matrix)
        # for each feature, 1 positive pair
        positive_pairs = torch.stack([matrix[i, i] for i in range(bs)], dim=0)
        # for each feature, (2B - 2) negative pairs
        negative_pairs = torch.stack([
            torch.stack([
                matrix[i, j]
                for j in range(bs * 2) if j != i and j != (i + bs)
            ], dim=0)
            for i in range(bs)
        ], dim=0)
        loss = -torch.log(positive_pairs / (negative_pairs.sum(dim=-1) + self.eps))
        return loss.mean()


class RefInfoNCELoss(nn.Module):
    """
    Basic idea from https://arxiv.org/pdf/2002.05709.pdf.
    Added a reference to decide how negative the pair is.
    """
    def __init__(self):
        super().__init__()
        self.cos_sim_func = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.eps = 1e-6
        return

    def forward(self, f, noise, ref):
        """
        For contrastive learning. Theoretical average min is e-3 (approx. -0.281718).
        Args:
            f (torch.Tensor): feature, shape of (B, L)
            noise (torch.Tensor): feature after agent task, shape of (B, L')
            ref (torch.Tensor): shape of (B, ...). to decide how bad the negative pair is.
                Instead of 1_[k!=j], use dist(ref_k, ref_j) to punish the negative pair,
                dist() == 1 means most negative pair, or sim() == min
                dist() == 0 means least negative pair, or sim() == max
        Returns:
            torch.Tensor:
        """
        bs = f.size()[0]
        cat_f_agent = torch.cat([noise, f], dim=0)
        # matrix: row i are 2bs pairs of f[i]: f0, f1, ..., a0, a1, ...
        matrix = torch.stack([self.cos_sim_func(f[i].repeat(bs * 2, 1), cat_f_agent)
                              for i in range(bs)], dim=0)
        matrix = torch.exp(matrix)
        ref = torch.cat([ref, ref], dim=0).unsqueeze(1)
        # for each feature, 1 positive pair
        positive_pairs = torch.stack([matrix[i, i] for i in range(bs)], dim=0)
        # for each feature, (2B - 2) negative pairs
        negative_pairs = torch.cat([
            torch.cat([
                torch.stack([
                    matrix[i, j] * (1.0 - self.cos_sim_func(ref[i], ref[j])) / 2.0
                    for j in range(bs * 2) if j != i and j != (i + bs)
                ], dim=1)
            ], dim=1)
            for i in range(bs)
        ], dim=0)
        loss = -torch.log(positive_pairs / (negative_pairs.sum(dim=-1) + self.eps))
        return loss.mean()


if __name__ == '__main__':
    f_ = torch.rand(8, 1000)
    agent_ = torch.rand(8, 1000)
    ref_ = torch.rand(8, 3)
    RefInfoNCELoss()(f_, agent_, ref_)

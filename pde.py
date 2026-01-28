from abc import abstractmethod

import torch
from torch import nn


class PDE(nn.Module):
    def __init__(
            self,
            linear=bool,
            gamma=torch.Tensor or None
    ):
        super(PDE, self).__init__()

        self.linear = linear
        if gamma is not None:
            self.register_buffer('gamma', gamma)

        self.first = True
        self.derivative_values = None
        self.source_values = None

    @abstractmethod
    def zeroth(self, x, u):
        """
        计算PDE零阶项
        Args:
            x: [N, 2] 坐标点
            u: [N, D] 预测值
        Returns:
            zeroth_terms: [N, D] 零阶项
        """
        pass

    @abstractmethod
    def derivative(self, x, u):
        """
        计算PDE导数项
        Args:
            x: [N, 2] 坐标点
            u: [N, D] 预测值
        Returns:
            derivative_terms: [N, D, 2, ...] 导数项，具体维度根据PDE定义而定
        """
        pass

    @abstractmethod
    def source(self, x):
        """
        计算PDE源项
        Args:
            x: [N, 2] 坐标点
        Returns:
            source_terms: [N, D] 源项
        """
        pass

    def reset(self):
        self.first = True
        self.derivative_values = None
        self.source_values = None

    def forward(self, x, u):
        if self.linear:
            if self.first:
                self.derivative_values = self.derivative(x, u)
                self.source_values = self.source(x)
            n_u = self.zeroth(x, u) + self.derivative_values - self.source_values

        else:
            n_u = self.zeroth(x, u) + self.derivative(x, u) - self.source(x)
        self.first = False
        return n_u

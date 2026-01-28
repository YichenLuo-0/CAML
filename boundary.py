from abc import abstractmethod
import torch
from torch import nn


class BoundaryCondition(nn.Module):
    def __init__(
            self,
            alpha: torch.Tensor,
            beta: torch.Tensor,
            normal_vector: torch.Tensor
    ):
        super().__init__()

        # 让它们跟着模型搬到 device / dtype
        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", beta)
        self.register_buffer("normal_vector", normal_vector)

        self.first = True
        self.derivative_values = None
        self.source_values = None

    def _split_points(
            self,
            tensor: torch.Tensor,
            mask: torch.Tensor
    ):
        # boundary points
        bc_idx = mask.bool().view(-1)  # [N_bc]
        tensor_boundary = tensor[bc_idx]
        return tensor_boundary

    def _mask_points(
            self,
            tensor: torch.Tensor,
            mask: torch.Tensor
    ):
        mask = mask.reshape(mask.shape[0], *([1] * (tensor.ndim - 1))).expand_as(tensor).float()
        tensor_boundary = tensor * mask
        return tensor_boundary

    def zeroth(self, u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # u: [N, C] where C in {1,2}
        # alpha: scalar or [C] or [1,C]
        u = self._split_points(u, mask)
        return self.alpha * u

    def derivative(self, x: torch.Tensor, u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [N, I]
        u: [N, C] where C in {1,2}
        normal_vector: [N, I]
        return: beta * du/dn -> [N, C]
        """

        # 先把非边界点屏蔽掉
        u_ = self._mask_points(u, mask)

        # 每个通道分别求梯度：得到 [N, C, I]
        grads = []
        for c in range(u_.shape[1]):
            du_c_dx = torch.autograd.grad(
                outputs=u_[:, c],  # [N]
                inputs=x,  # [N, I]
                grad_outputs=torch.ones_like(u_[:, c]),
                create_graph=True,
                retain_graph=True,  # 多通道需要保留图
                only_inputs=True
            )[0]  # [N, I]
            grads.append(du_c_dx)

        du_dx = torch.stack(grads, dim=1)  # [N, C, I]
        du_dx = self._split_points(du_dx, mask)

        # 法向导数：sum_i n_i * du/dx_i -> [N, C]
        normal_derivative = torch.einsum("ni,nci->nc", self.normal_vector, du_dx)

        # beta 广播到 [N, C]
        return self.beta * normal_derivative

    @abstractmethod
    def source(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        应返回 [N, C]，与 u 的通道数一致（C=1 或 2）
        """
        pass

    def reset(self):
        self.first = True
        self.derivative_values = None
        self.source_values = None

    def forward(self, x: torch.Tensor, u: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.first:
            self.derivative_values = self.derivative(x, u, mask)
            self.source_values = self.source(x, mask)
        res = self.zeroth(u, mask) + self.derivative_values - self.source_values
        self.first = False
        return res

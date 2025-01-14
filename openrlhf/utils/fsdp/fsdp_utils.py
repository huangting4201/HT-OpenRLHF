import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import torch
from torch import Tensor
import torch.distributed as dist
from torch.optim import Optimizer

from openrlhf.utils.utils import get_current_device
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

try:
    import amp_C
    from apex.multi_tensor_apply import multi_tensor_applier

    APEX_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    logger.warning("The torch implementation for cal_l2norm is slower than apex. Please note this!")
    APEX_AVAILABLE = False

inf = math.inf


def multi_tensor_l2norm_torch(tensor_list, per_tensor):
    # Convert tensor_list elements to torch.float32
    tensor_list = [tensor.float() for tensor in tensor_list]
    norms_tensor = torch.stack([torch.norm(tensor, p=2) for tensor in tensor_list])
    l2_norm = torch.norm(norms_tensor, p=2).unsqueeze(0)

    if per_tensor:
        per_tensor_norm = norms_tensor
    else:
        per_tensor_norm = torch.Tensor([]).to(norms_tensor.device)

    return l2_norm, per_tensor_norm


def calc_l2_norm(grads):
    norm = 0.0
    if len(grads) > 0:
        if APEX_AVAILABLE:
            dummy_overflow_buf = torch.tensor([0], device=get_current_device(), dtype=torch.int32)
            norm, _ = multi_tensor_applier(
                amp_C.multi_tensor_l2norm,
                dummy_overflow_buf,
                [grads],
                False,  # no per-parameter norm
            )
        else:
            norm, _ = multi_tensor_l2norm_torch(grads, False)
    return norm


def calc_lp(grads, norm_type):
    norm = 0.0
    for grad in grads:
        grad_norm = torch.norm(grad, norm_type)
        norm += grad_norm**norm_type
    return norm


def get_norm(grads, norm_type, enable_cuda_kernels):
    if norm_type == inf:
        grad_norm = max(g.data.abs().max() for g in grads)
    elif norm_type == 2.0 and enable_cuda_kernels:
        grad_norm = calc_l2_norm(grads) ** norm_type
    else:
        grad_norm = calc_lp(grads, norm_type)
    return grad_norm


def reduce_grads(gradients, parameters):
    parallel_grads = []

    for g, _ in zip(gradients, parameters):
        # process all ranks for FSDP parameter group
        parallel_grads.append(g.data.float())

    return parallel_grads


def get_tensor_norm(norm: Union[float, torch.Tensor], move_to_cuda) -> torch.Tensor:
    if isinstance(norm, float):
        norm = torch.Tensor([norm])
    if move_to_cuda:
        norm = norm.to(get_current_device())
    return norm


def compute_norm(gradients, parameters, norm_type=2):
    """Get the norm
    Arguments:
        gradients (Iterable[Tensor]): The gradient value.
        parameters (Iterable[Tensor]): The parameter each gradient corresponds to.
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters, need total_norm**(1/norm) before using.
    """

    enable_cuda_kernels = gradients[0].device.type != "cpu"
    # Norm parameters.
    norm_type = float(norm_type)

    tensor_parallel_grads = reduce_grads(gradients, parameters)
    tensor_parallel_norm = get_norm(tensor_parallel_grads, norm_type, enable_cuda_kernels)

    # If norm is type of float, then we convert them into torch.Tensor.
    tensor_parallel_norm = get_tensor_norm(tensor_parallel_norm, enable_cuda_kernels)
    # If grads are on CPU, the norms is also on CPU. Cast them to CUDA tensors
    if not enable_cuda_kernels:
        tensor_parallel_norm = tensor_parallel_norm.to(get_current_device())

    total_norm = tensor_parallel_norm

    """
    Sum across all model-parallel GPUs.
    """
    dist.all_reduce(total_norm, op=dist.ReduceOp.SUM)

    if torch.is_tensor(total_norm):
        total_norm = total_norm.item()

    # Scale.
    if total_norm == float("inf") or total_norm == -float("inf"):
        total_norm = -1

    if math.isnan(total_norm):
        total_norm = -2

    return total_norm


class BaseGradScaler(ABC):
    """A base class for the gradient scaler.

    Args:
        initial_scale (float): the initial loss scale
    """

    def __init__(self, initial_scale: float):
        assert initial_scale > 0
        self._scale = torch.tensor([initial_scale], device=get_current_device(), dtype=torch.float32)

    @property
    def scale(self) -> Tensor:
        """Returns the loss scale."""

        return self._scale

    @property
    def inv_scale(self) -> Tensor:
        """Returns the inverse of the loss scale."""

        return self._scale.double().reciprocal().float()

    def state_dict(self) -> Dict:
        """Returns the states of the gradient scaler as a dict object."""

        state_dict = dict()
        state_dict["scale"] = self.scale
        return state_dict

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load the states of the gradient scaler from a dict object.

        Args:
            state_dict (dict): the states of the gradient scaler
        """

        self._scale = state_dict["scale"]

    @abstractmethod
    def update(self, overflow: bool) -> None:
        """Update the loss scale.

        Args:
            overflow (bool): whether overflow occurs
        """

        pass


class DynamicGradScaler(BaseGradScaler):
    """A gradient scaler which uses dynamic loss scale

    Args:
        initial_scale (float): the initial loss scale, defaults to 2**16
        growth_factor (float): the multiplication factor for increasing loss scale, defaults to 2
        backoff_factor (float): the multiplication factor for decreasing loss scale, defaults to 0.5
        growth_interval (int): the number of steps to increase loss scale when no overflow occurs, defaults to 1000
        min_scale (float): the minimum loss scale, defaults to 1
        max_scale (float): the maximum loss scale, defaults to 2**24
        hysteresis (int):  the number of overflows before decreasing loss scale, defaults to 2
    """

    def __init__(
        self,
        initial_scale: float = 2**16,
        growth_factor: float = 2,
        backoff_factor: float = 0.5,
        growth_interval: int = 1000,
        min_scale: Optional[float] = 1,
        max_scale: Optional[float] = 2**24,
        hysteresis: int = 2,
        dtype=torch.bfloat16,
    ):
        super().__init__(initial_scale)
        if min_scale:
            self._min_scale = torch.tensor([min_scale], device=get_current_device(), dtype=torch.float32)
        else:
            self._min_scale = None

        if max_scale:
            self._max_scale = torch.tensor([max_scale], device=get_current_device(), dtype=torch.float32)
        else:
            self._max_scale = None

        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._growth_step = 0
        self._hysteresis = hysteresis
        self._hysteresis_step = 0
        self._dtype = dtype
        self._sanity_checks()

    def _sanity_checks(self) -> None:
        """Check if the arguments are correct."""

        assert self._dtype in [torch.float16, torch.bfloat16, torch.float32]

        if self._min_scale is not None:
            min_scale = self._min_scale.item()
            assert min_scale > 0, "The minimum gradient scale cannot be zero or negative"

            if self._dtype != torch.float16 and min_scale != 1.0:
                logger.warning(f"Detect you use {self._dtype}, but min_scale: {min_scale} != 1.0")

        if self._max_scale:
            max_scale = self._max_scale.item()
            assert max_scale > 0, "The maximum gradient scale cannot be zero or negative"

            if self._dtype != torch.float16 and max_scale != 1.0:
                logger.warning(f"Detect you use {self._dtype}, but max_scale: {max_scale} != 1.0")

        if self._dtype == torch.float16:
            assert self._growth_factor > 1.0, "The growth factor cannot be equal or smaller than 1"
            assert self._backoff_factor < 1.0 and self._backoff_factor > 0, "The backoff factor must be between 0 and 1"
        else:
            assert self._growth_factor >= 1.0, "The growth factor cannot be smaller than 1"
            assert (
                self._backoff_factor <= 1.0 and self._backoff_factor > 0
            ), "The backoff factor must be between 0 and 1"

            if self._growth_factor != 1.0:
                logger.warning(f"Detect you use {self._dtype}, but growth_factor: {self._growth_factor} != 1.0")
            if self._backoff_factor != 1.0:
                logger.warning(f"Detect you use {self._dtype}, but backoff_factor: {self._backoff_factor} != 1.0")

        assert self._hysteresis >= 0, "The hysteresis cannot be negative"

    def update(self, overflow: bool) -> None:
        """Update the loss scale.

        Args:
            overflow (bool): whether overflow occurs
        """
        if overflow:
            self._hysteresis_step += 1
            self._growth_step = 0

            if self._hysteresis_step >= self._hysteresis:
                self._backoff_scale()
                logger.warning(f"Overflow occurs, the loss scale is adjusted to {self.scale.item()}")
        else:
            self._growth_step += 1
            if self._growth_step == self._growth_interval:
                self._growth_step = 0
                self._hysteresis_step = 0
                self._grow_scale()
                logger.warning(
                    f"No overflow for consecutive {self._growth_interval} steps, "
                    f"the loss scale is adjusted to {self.scale.item()}",
                )

    def _backoff_scale(self) -> None:
        """Decrease the loss scale"""

        self._scale = self._scale * self._backoff_factor
        if self._min_scale:
            self._scale = torch.max(self._scale, self._min_scale)

    def _grow_scale(self) -> None:
        """Increase the loss scale"""

        self._scale = self._scale * self._growth_factor
        if self._max_scale:
            self._scale = torch.min(self._scale, self._max_scale)

    def state_dict(self):
        """Returns the states of the gradient scaler as a dict object."""

        state_dict = dict()
        state_dict["_scale"] = self._scale.item()
        state_dict["_growth_step"] = self._growth_step
        state_dict["_hysteresis_step"] = self._hysteresis_step

        return state_dict

    def load_state_dict(self, state_dict):
        """Load the states of the gradient scaler from a dict object.

        Args:
            state_dict (dict): the states of the gradient scaler
        """

        self._scale = self._scale.fill_(state_dict["_scale"])
        self._growth_step = state_dict["_growth_step"]
        self._hysteresis_step = state_dict["_hysteresis_step"]


class BaseOptimizer(Optimizer):
    """
    Base Optimizer.
    """

    def __init__(self, optim: Optimizer):  # pylint: disable=W0231
        self.optim = optim

    @property
    def param_groups(self):
        return self.optim.param_groups

    @property
    def defaults(self):
        return self.optim.defaults

    def add_param_group(self, *args, **kwargs):
        return self.optim.add_param_group(*args, **kwargs)

    def step(self, *args, **kwargs):
        return self.optim.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        self.optim.zero_grad(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.optim.load_state_dict(*args, **kwargs)

    def state_dict(self):
        return self.optim.state_dict()

    def backward(self, loss):
        loss.backward()

    def backward_by_grad(self, tensor, grad):
        torch.autograd.backward(tensors=tensor, grad_tensors=grad)

    def clip_grad_norm(self):
        pass

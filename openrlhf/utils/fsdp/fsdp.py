import functools
import os
import pickle
import random
import shutil
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from peft import PeftModel, get_peft_model_state_dict
from torch import distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed._shard.api import load_with_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.api import BackwardPrefetch, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaModel, LlamaForCausalLM

from openrlhf.models import Actor
from openrlhf.models.ring_attn_utils import get_ring_attn_group, set_ring_attn_group
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.utils.deepspeed.deepspeed_utils import (
    _z3_params_to_fetch,
    get_optimizer_grouped_parameters,
)
from openrlhf.utils.utils import get_current_device


ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]


def get_module_class_from_name(module, name):
    """
    Gets a class from a module by its name.

    Args:
        module (`torch.nn.Module`): The module to get the class from.
        name (`str`): The name of the class.
    """
    modules_children = list(module.children())
    if module.__class__.__name__ == name:
        return module.__class__
    elif len(modules_children) == 0:
        return
    else:
        for child_module in modules_children:
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class


class FSDPStrategy(ABC):
    """
    The strategy for training with Accelerator FSDP.
    """

    def __init__(
        self,
        seed: int = 42,
        max_norm: float = 0.0,
        micro_train_batch_size=1,
        train_batch_size=1,
        mode="v1",
        bf16=True,
        args=None,
    ) -> None:
        super().__init__()

        self.args = args
        self.mode = mode
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size = micro_train_batch_size
        self.bf16 = bf16
        self.seed = seed
        self.max_norm = max_norm
        self.adam_offload = getattr(args, "adam_offload", False)
        self.zpg = getattr(args, "zpg", 1)
        self.grad_accum_dtype = getattr(args, "grad_accum_dtype", None)
        # overlap_comm
        self.overlap_comm = getattr(args, "overlap_comm", False)

        self.is_rlhf = False
        self.time_steps = defaultdict(int)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_distributed(self, timeout=timedelta(minutes=30)) -> None:
        self.set_seed(self.seed)

        if self.args.local_rank == -1 and "LOCAL_RANK" in os.environ:  # for slurm
            self.args.local_rank = int(os.environ["LOCAL_RANK"])

        if self.args.local_rank != -1:
            torch.cuda.set_device(self.args.local_rank)

        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        try:
            rank = int(os.environ["RANK"])
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            host = os.environ["MASTER_ADDR"]
            port = int(os.environ["MASTER_PORT"])
        except KeyError as e:
            raise RuntimeError(f"Could not find {e} in the torch environment")

        # initialize the default process group
        init_method = f"tcp://[{host}]:{port}"
        dist.init_process_group(
            rank=rank,
            world_size=world_size,
            backend="nccl",
            init_method=init_method,
        )

        self.setup_ring_attn()
        self.world_size = dist.get_world_size()
        print(f"Init Distributed Env, world_size:{self.world_size}, local_rank:{dist.get_rank()}", flush=True)
        self.accumulated_gradient = (
            self.train_batch_size * self.ring_attn_size // self.micro_train_batch_size // self.world_size
        )

    def setup_ring_attn(self):
        self.ring_attn_size = getattr(self.args, "ring_attn_size", 1)
        if self.ring_attn_size == 1:
            self.ring_attn_rank = 0
            return

        ring_head_stride = getattr(self.args, "ring_head_stride", 1)
        for i in range(dist.get_world_size() // self.ring_attn_size):
            ring_attn_ranks = list(
                range(
                    i * self.ring_attn_size,
                    (i + 1) * self.ring_attn_size,
                )
            )
            group = dist.new_group(ranks=ring_attn_ranks, backend="nccl")
            if dist.get_rank() in ring_attn_ranks:
                set_ring_attn_group(group)
                self.ring_attn_rank = dist.get_rank(group=group)

        from ring_flash_attn import substitute_hf_flash_attn

        substitute_hf_flash_attn(self.ring_attn_group, ring_head_stride)

    @property
    def ring_attn_group(self):
        return get_ring_attn_group()

    def create_optimizer(self, model, **kwargs) -> Optimizer:
        if isinstance(model, Actor):
            model = model.model
        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.adam_offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(model, kwargs["weight_decay"])
        optim = AdamOptimizer(optim_params, **kwargs)
        return optim

    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        # if isinstance(model, Actor):
        #     model = model.model
        # model.backward(loss)
        loss.backward()

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler,
        name="model",
        **kwargs,
    ) -> None:
        # if isinstance(model, Actor):
        #     model = model.model
        # model.step()
        optimizer.step()
        scheduler.step()

    def setup_dataloader(
        self,
        replay_buffer,
        batch_size: int,
        pin_memory: bool = False,
        shuffle=True,
        collate_fn=None,
        drop_last=True,
        sampler=None,
        consumed_samples=0,
    ):
        # DDP only mode, replay buffers on each rank are different.
        if sampler is None:
            num_replicas = dist.get_world_size() // self.ring_attn_size
            rank = dist.get_rank() // self.ring_attn_size
            sampler = DistributedSampler(
                replay_buffer,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                seed=self.seed,
                drop_last=drop_last,
                consumed_samples=consumed_samples,
            )

        return DataLoader(
            replay_buffer,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

    def _unwrap_model(self, model) -> nn.Module:
        if isinstance(model, Actor):
            return self._unwrap_model(model.model)
        elif hasattr(model, "module"):
            return model.module
        else:
            return model

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair, is_rlhf=False
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        ret = []
        self.is_rlhf = is_rlhf
        for arg in models_or_model_optim_pairs:
            assert not isinstance(arg, tuple)
            ret.append(self._fsdp1_init_model(arg))

        return ret[0] if len(ret) == 1 else ret

    def _fsdp1_init_model(self, model):
        is_actor = isinstance(model, Actor)

        # transformer_cls_names_to_wrap = (
        #     ["LlamaForCausalLM", "LlamaModel", "LlamaDecoderLayer"] if is_actor else ["LlamaModel", "LlamaDecoderLayer"]
        # )
        # transformer_cls_names_to_wrap = ["LlamaModel", "LlamaDecoderLayer"]
        # transformer_cls_to_wrap = set()
        # for layer_class in transformer_cls_names_to_wrap:
        #     transformer_cls = get_module_class_from_name(model, layer_class)
        #     if transformer_cls is None:
        #         raise ValueError(f"Could not find the transformer layer class {layer_class} in the model.")
        #     transformer_cls_to_wrap.add(transformer_cls)

        transformer_cls_to_wrap = [LlamaModel, LlamaDecoderLayer]
        sharded_model = FSDP(
            module=model.model if is_actor else model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO2: SHARD_GRAD_OP, ZeRO3: FULL_SHARD
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy, transformer_layer_cls=transformer_cls_to_wrap
            ),
            forward_prefetch=True,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
            use_orig_params=True,
            device_id=get_current_device(),
        )

        if self.is_rank_0():
            print(f"after _fsdp1_init_model: {sharded_model}", flush=True)

        if is_actor:
            model.model = sharded_model
        else:
            model = sharded_model

        # model = sharded_model

        return model.to(device=get_current_device())

    def _fsdp2_init_model(self, model):
        is_actor = isinstance(model, Actor)

        from torch.distributed import init_device_mesh
        from torch.distributed._composable.fsdp import fully_shard

        # world_mesh = init_device_mesh(device_type="cuda", mesh_shape=(8, 1), mesh_dim_names=("shard", "replica"))
        # dp_mesh = world_mesh["replica"]
        fsdp_kwargs = {
            # "mesh": dp_mesh,
            "reshard_after_forward": True,  # ZeRO2: False, ZeRO3: True
        }

        for module in model.modules():
            if isinstance(module, LlamaDecoderLayer):
                fully_shard(module, **fsdp_kwargs)

        # for module in model.modules():
        #     if isinstance(module, LlamaModel):
        #         fully_shard(module, **fsdp_kwargs)

        # for module in model.modules():
        #     if isinstance(module, LlamaForCausalLM):
        #         fully_shard(module, **fsdp_kwargs)

        fully_shard(model if is_actor else model, **fsdp_kwargs)

        if self.is_rank_0():
            print(f"after _fsdp2_init_model: {model}", flush=True)

        return model.to(device=get_current_device())

    def moving_average(self, model, model_ema, beta=0.992, device="cpu"):
        self.time_steps["ema"] += 1
        if self.time_steps["ema"] % self.accumulated_gradient == 0:
            with torch.no_grad():
                for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                    if param.requires_grad:
                        if self.stage != 3:
                            data = param.data.to(device)
                            param_ema.data.copy_((1 - beta) * data + beta * param_ema.data)
                        else:
                            # TODO: use prefiltering for efficiency
                            params_to_fetch = _z3_params_to_fetch([param, param_ema])
                            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                                data = param.data.to(device)
                                param_ema.data.copy_((1 - beta) * data + beta * param_ema.data)

    def load_model(
        self,
        model: nn.Module,
        path: str,
        map_location="cpu",
        strict: bool = False,
        key_replace_fn=None,
    ) -> None:
        # unwrapped_model = self._unwrap_model(model)
        is_actor = isinstance(model, Actor)
        assert is_actor
        model_to_load = model.model if is_actor else model
        state_dict = torch.load(path, map_location=map_location)
        if key_replace_fn:
            state_dict = key_replace_fn(state_dict)
        model_to_load.load_state_dict(state_dict, strict=strict)

    def save_model(self, model: nn.Module, tokenizer, output_dir, **kwargs) -> None:
        # save model weights for ZeRO2/3
        # model_to_save = self._unwrap_model(model)
        is_actor = isinstance(model, Actor)
        assert is_actor
        model_to_save = model.model if is_actor else model

        # import pdb

        # pdb.set_trace()

        full_states = get_model_state_dict(
            model_to_save, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        assert full_states is not None

        if self.is_rank_0():

            print(f"ht debug model_to_save:{model_to_save}", flush=True)

            os.makedirs(output_dir, exist_ok=True)

            with torch.device(get_current_device()):
                total_model = AutoModelForCausalLM.from_pretrained(
                    self.args.pretrain,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2" if self.args.flash_attn else "eager",
                    quantization_config=None,
                    torch_dtype=torch.bfloat16 if self.args.bf16 else "auto",
                    device_map=None,
                )

            total_model.load_state_dict(full_states, strict=True, assign=True)

            print(f"ht debug total_model:{total_model}", flush=True)
            total_model.save_pretrained(save_directory=output_dir, safe_serialization=True)

            # save config
            output_config_file = os.path.join(output_dir, "config.json")
            model_to_save.config.to_json_file(output_config_file)
            # save tokenizer
            tokenizer.save_pretrained(output_dir)

            # for models not in AutoModel, copy python module files
            train_from_model_path = model_to_save.config._name_or_path
            if os.path.exists(train_from_model_path):
                for filename in os.listdir(train_from_model_path):
                    if filename.endswith(".py"):
                        shutil.copy(os.path.join(train_from_model_path, filename), os.path.join(output_dir, filename))

        dist.barrier()

    def all_reduce(self, data, op="mean"):
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"

            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())
            if op == "mean":
                data /= self.world_size
            dist.all_reduce(data, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data

    def all_gather(self, data):
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"

            ret = [torch.zeros_like(data).to(torch.cuda.current_device()) for _ in range(self.world_size)]
            dist.all_gather(ret, data.to(torch.cuda.current_device()))
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)

    def print(self, *msg):
        if self.is_rank_0():
            print(*msg)

    def is_rank_0(self) -> bool:
        return dist.get_rank() == 0

    def get_rank(self) -> int:
        return dist.get_rank()

    def save_ckpt(self, model, save_dir, tag=None, max_num=3, max_mem=1000, client_state={}, save_latest=True):
        if self.is_rank_0():
            os.makedirs(f"{save_dir}/{tag}", exist_ok=True)
            MAX_SIZE = max_mem * 1024**3  # Convert GB to bytes

            while True:
                subdirs = sorted(
                    [
                        (os.path.join(save_dir, d), os.path.getmtime(os.path.join(save_dir, d)))
                        for d in os.listdir(save_dir)
                        if os.path.isdir(os.path.join(save_dir, d))
                    ],
                    key=lambda x: x[1],
                )
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, f))
                    for subdir, _ in subdirs
                    for dirpath, _, filenames in os.walk(subdir)
                    for f in filenames
                )

                if len(subdirs) >= max_num or total_size > MAX_SIZE:
                    oldest_dir = subdirs[0][0]
                    if os.path.exists(oldest_dir):
                        shutil.rmtree(oldest_dir)
                        self.print(f"Deleted oldest ckpt {oldest_dir}")
                else:
                    break

        dist.barrier()

        # FSDP model can only save with sharded shape SHARDED_STATE_DICT when set use_orig_params=True
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            shard_states = model.state_dict()

        fn = f"{tag}/model_rank{dist.get_rank()}.pt"
        fp = os.path.join(save_dir, fn)
        torch.save(shard_states, fp, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def load_ckpt(
        self,
        model,
        load_dir,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
    ):
        load_dir = "ckpt/checkpoints_dpo/global_step120"
        should_load_name = f"model_rank{dist.get_rank()}.pt"
        fp = os.path.join(load_dir, should_load_name)

        # for FSDP shards loading, we need to set process group
        # with load_with_process_group():
        states = torch.load(fp, map_location=get_current_device())

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            missing_k, unexpected_keys = model.load_state_dict(states, strict=load_module_strict)

        # avoid to cuda oom, Ref: https://discuss.pytorch.org/t/load-state-dict-causes-memory-leak/36189/11
        # del states

        dist.barrier()

        return load_dir, states

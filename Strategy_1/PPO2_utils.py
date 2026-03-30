import torch
import torch.nn as nn

import random
from collections import namedtuple

import numpy as np
'''
此文件部分内容copy from TD-MPC2 common文件夹下 layers.py
'''
class NormedLinear(nn.Linear):

    def __init__(self, *args, act=None, dropout=0.,  layernorm=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features) if layernorm else nn.Identity()
        if act is None:
            act = nn.Mish(inplace=False)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return f"NormedLinear(in_features={self.in_features}, " \
               f"out_features={self.out_features}, " \
               f"bias={self.bias is not None}{repr_dropout}, " \
               f"act={self.act.__class__.__name__})"


def mlp(in_dim, mlp_dims, out_dim, act=None ,out_act=None, dropout=0., layernorm=True):
    """
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations, and optionally dropout.
	"""
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = nn.ModuleList()
    for i in range(len(dims) - 2):
        mlp.append(NormedLinear(dims[i], dims[i + 1], act=act, dropout=dropout * (i == 0),layernorm = layernorm))
    mlp.append(NormedLinear(dims[-2], dims[-1], act=out_act) if out_act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*mlp)


class Memory(object):
    # --- 2. 修改 __init__ 方法以接收字段定义 ---
    def __init__(self, fields=('state', 'action', 'next_state', 'mask')):
        # 将 Trajectory 定义为实例属性
        self.Trajectory = namedtuple('Trajectory', fields)
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(self.Trajectory(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return self.Trajectory(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return self.Trajectory(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)
    

class gymEnvWrapper:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        # 通过 @reproducible.setter 装饰器触发 setter 逻辑
        self.reproducible = config['env_reproducible']
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get('use_gpu', False) else "cpu")

    @property
    def reproducible(self):
        """获取可复现性状态"""
        return self._reproducible

    @reproducible.setter
    def reproducible(self, value: bool):
        """
        设置可复现性状态。
        当这个属性被赋值时，此方法会自动被调用。
        """
        self._reproducible = value
        self.seed = self.config['seed'] if self._reproducible else None
        #print(f"Environment reproducibility set to {self._reproducible}. Seed is now: {self.seed}")


    def reset(self,): ## TODO
        state = self.env.reset(seed = self.seed)[0]
        # 归一state 到[-1,1]
        state = 2 * (state - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low) - 1
        return torch.as_tensor(state, dtype=torch.float32).to(self.device)

    def step(self, state, action):
        with torch.no_grad(): # 如果是环境模型有梯度，则要加上这个
            # 反归一action
            action = np.clip(action.cpu().numpy(), -1.0, 1.0)
            action = action * (self.env.action_space.high - self.env.action_space.low) / 2 + (self.env.action_space.high + self.env.action_space.low) / 2

            next_state, reward, terminate, truncation, info = self.env.step(action)
            done = terminate or truncation

            info['terminate'] = torch.as_tensor(terminate, dtype=torch.float32).to(self.device)

            next_state = 2 * (next_state - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low) - 1
            return (torch.as_tensor(next_state, dtype=torch.float32).to(self.device),
                    torch.as_tensor(reward, dtype=torch.float32).to(self.device),
                    torch.as_tensor(done, dtype=torch.float32).to(self.device),
                    info)
        

class gymEnvWrapper_for_env_stock:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.env_action_num = self.env.action_num
        # 通过 @reproducible.setter 装饰器触发 setter 逻辑
        self.reproducible = config['env_reproducible']
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get('use_gpu', False) else "cpu")

    @property
    def reproducible(self):
        """获取可复现性状态"""
        return self._reproducible

    @reproducible.setter
    def reproducible(self, value: bool):
        """
        设置可复现性状态。
        当这个属性被赋值时，此方法会自动被调用。
        """
        self._reproducible = value
        self.seed = self.config['seed'] if self._reproducible else None
        #print(f"Environment reproducibility set to {self._reproducible}. Seed is now: {self.seed}")


    def reset(self,seed=None): ## TODO
        current_seed = seed if seed is not None else self.seed
        state = self.env.reset(seed = current_seed)[0]
        # 归一state 到[-1,1]
        #state = 2 * (state - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low) - 1
        return torch.as_tensor(state, dtype=torch.float32).to(self.device)

    def step(self, state, action):
        with torch.no_grad(): # 如果是环境模型有梯度，则要加上这个
            # # 反归一action
            # action = np.clip(action.cpu().numpy(), -1.0, 1.0)
            # ## 根据比例拓展到8个动作
            # action_idx = int((action + 1) * 4)
            # action = min(action_idx, 7)

            if self.config.get('discrete', False):
                # action 可能是 tensor 或 python int
                if isinstance(action, torch.Tensor):
                    # 可能是形如 tensor(1) 或 tensor([1])
                    action_np = int(action.cpu().numpy().squeeze())
                else:
                    action_np = int(action)
                next_state, reward, terminate, truncation, info = self.env.step(action_np)

            else:
                # 1. 确保输入在 [-1.0, 1.0] 范围内
                action_val = np.clip(action.cpu().numpy(), -1.0, 1.0)
                
                # 2. 通用映射逻辑：
                # 将 [-1, 1] 线性映射到 [0, env_action_num - 1]
                # 计算方式：((x + 1) / 2) * (num - 1)
                action_idx = int(((action_val + 1.0) / 2.0) * (self.env_action_num))
                
                # 3. 再次确保不越界
                action = min(action_idx, self.env_action_num - 1)

                #action = action * (self.env.action_space.high - self.env.action_space.low) / 2 + (self.env.action_space.high + self.env.action_space.low) / 2

                next_state, reward, terminate, truncation, info = self.env.step(action)
            done = terminate or truncation

            info['terminate'] = torch.as_tensor(terminate, dtype=torch.float32).to(self.device)

            #next_state = 2 * (next_state - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low) - 1
            return (torch.as_tensor(next_state, dtype=torch.float32).to(self.device),
                    torch.as_tensor(reward, dtype=torch.float32).to(self.device),
                    torch.as_tensor(done, dtype=torch.float32).to(self.device),
                    info)
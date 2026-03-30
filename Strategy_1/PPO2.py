import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import time
import numpy as np
from torch import nn
import copy
import pickle 
from PPO2_utils import  mlp, Memory ,gymEnvWrapper_for_env_stock
from torch.distributions import Normal, Categorical

from config import PPO_COBFIG
from env_etf import ETFTradingEnv
import pandas as pd

class Actor(nn.Module):
    def __init__(self, config, s_dim: int, hidden_dims ,a_dim: int, act=None, out_act=None, dropout=0.,layernorm=True):
        super().__init__()
        self.config = config
        self.backbone = mlp(s_dim, hidden_dims, a_dim, act=act ,out_act=out_act, dropout=dropout,layernorm=layernorm)

        self.log_std = nn.Parameter(torch.zeros(1, a_dim))

    def forward(self, s: torch.Tensor):
        # 这一步等价于原来的 l1、l2、mean_layer 连在一起
        mean = self.backbone(s)
        mean = torch.tanh(mean)  # 把动作均值压到 [-1, 1]
        if s.dim() == 1:
            log_std = self.log_std.squeeze(0)
        else:
            log_std = self.log_std.expand_as(mean)
        # log_std 仍然是单独的参数，只在这里展开
        #log_std = self.log_std.expand_as(mean)
        log_std = torch.clamp(log_std, self.config['PPO']['log_std_min'], self.config['PPO']['log_std_max'])
        std = torch.exp(log_std)
        return mean, std

# 新增：离散动作用的 Actor（输出 logits）
class ActorDiscrete(nn.Module):
    def __init__(self, config, s_dim: int, hidden_dims, a_dim: int):
        super().__init__()
        self.config = config
        # 输出维度为 a_dim (logits)
        self.backbone = mlp(s_dim, hidden_dims, a_dim, act=getattr(nn, config['PPO']['activation'])(), out_act=None,
                            dropout=config['PPO']['dropout'], layernorm=config['PPO']['layernorm'])

    def forward(self, s: torch.Tensor):
        logits = self.backbone(s)  # no tanh
        return logits


class Critic(nn.Module):
    def __init__(self, config, s_dim: int, hidden_dims ,a_dim: int, act=None, out_act=None, dropout=0.,layernorm=True):
        super().__init__()
        self.config = config
        self.backbone = mlp(s_dim, hidden_dims, a_dim, act=act ,out_act=out_act, dropout=dropout,layernorm=layernorm)

    def forward(self, s: torch.Tensor):
        v_out = self.backbone(s)
        return v_out

    
class PolicyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        act = getattr(nn, config['PPO']['activation'])()
        # 支持离散/连续分支
        if self.config.get('discrete', False):
            self.P = ActorDiscrete(config, config['s_dim'], 2 * [config['PPO']['policy_mlp_dim']], config['a_dim'])
        else:
            self.P = Actor(config, config['s_dim'], 2 * [config['PPO']['policy_mlp_dim']], config['a_dim'],
                           act=act, dropout=config['PPO']['dropout'], layernorm=config['PPO']['layernorm'])
        self.V = Critic(config, config['s_dim'], 2 * [config['PPO']['policy_mlp_dim']], 1,
                        act=act, dropout=config['PPO']['dropout'], layernorm=config['PPO']['layernorm'])

    def pi(self, s: torch.Tensor, return_dist: bool = False, return_mean: bool = False):
        ''' 策略网络前向传播，支持离散/连续 '''
        if self.config.get('discrete', False):
            logits = self.P(s)
            if return_mean:
                # --- 修改：如果是评估/确定性测试，直接返回概率最大的动作索引 ---
                return torch.argmax(logits, dim=-1)
            dist = Categorical(logits=logits)
            if return_dist:
                return dist
            action = dist.sample()  # shape: [batch] (LongTensor)
            return action
        else:
            mean, std = self.P(s)
            if return_mean:
                return mean
            dist = Normal(mean, std)
            if return_dist:
                return dist
            action = dist.sample()
            return action
    
    
    def value(self, s: torch.Tensor):
        v_out = self.V(s)
        return v_out
    
class PPO:
    def __init__(self, config):
        self.global_seed(config['seed'])
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get('use_gpu', False) else "cpu")
        
        self.P_model = PolicyModel(config).to(self.device)
        self.P = self.P_model.P
        self.V = self.P_model.V
        self.pi = self.P_model.pi
        self.value = self.P_model.value
        

        betas = [0.9, 0.999]

        self.optim = torch.optim.Adam(
            [
                {'params': self.P.parameters() , 'lr': config['PPO']['p_lr'] , 'betas' : betas},
                {'params': self.V.parameters(), 'lr': config['PPO']['v_lr'] ,'betas' : betas},
            ]
        )

    def global_seed(self, seed: int):
        ''' 设置随机种子 '''
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    

    def load_model(self, path: str):
        checkpoint = torch.load(path,weights_only=False)
        ## 解析存储地址
        self.config['log_dir'] = os.path.dirname(path)

        self.P.load_state_dict(checkpoint['P_state_dict'])
        self.V.load_state_dict(checkpoint['V_state_dict'])
        self.optim.load_state_dict(checkpoint['optim_state_dict'])
        torch.set_rng_state(checkpoint['cpu_rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])

        iteration = checkpoint.get('iteration', 0)
        return iteration

    def save_model(self, path: str = None,iteration: int =0, name='',for_pkl=False):
        '''
        iteration: 
            'best': 当前训练的episode数 以0开始
            'final': 总训练episode数 = 之后读取时要继续训练的开始episode数
        '''

        name = f'_{name}' if name != '' else ''
        self.model_pth = os.path.join(self.config["log_dir"] , f'model{name}.pth')
        path = path if path is not None else self.model_pth 

        P_model = copy.deepcopy(self.P_model).to('cpu')
        
        state_dict = {
            'iteration': iteration,  # <--- 新增：保存当前步数
            'P_state_dict': P_model.P.state_dict(),
            'V_state_dict': P_model.V.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'cpu_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state()
        }
        torch.save(state_dict, path)
        
        # 导出为pkl
        if for_pkl:
            try:
                #current_time = time.strftime("%Y%m%d%H%M%S")
                pkl_output_filename = f'model{name}.pkl'
                pkl_output_path = os.path.join(self.config["log_dir"], pkl_output_filename)
                export_dict = {'P_state_dict': state_dict['P_state_dict']}
                with open(pkl_output_path, 'wb') as f:
                    pickle.dump(export_dict, f)
            except Exception as e:
                print(f"❌ P_state_dict 导出失败: {str(e)}")

        del P_model

    
    def GAE(self,rewards, values, masks_1, masks_2, gamma, lam, last_value=None, MC=False,):
        ''' 计算GAE优势函数 
        一般来说
        mask_1 (td_delta) 表示 1-terminate
        mask_2 (gae)     表示  1-(terminate or truncation)
        令done = terminate or truncation 
        truncation 由这里的dataset['index'] 提供

        如果不提供 terminate 则都认为是0, 即 masks_1 全是1  # 这里不提供 若要提供 则需要另外学习这个terminate函数
        如果不提供 last_value 则可以写成是tensor(0)  # 这里逻辑提供了last_value

        return 
        {
            1.advantages 来更新P           使用GAE(λ)
            2.returns    来更新V           这里写两种更新V的方式1.TD(λ) 2.Monte-Carlo(TD(1))
        }
        '''
        if last_value is None:
            last_value = torch.tensor(0.0).to(self.device)

        with torch.no_grad():
            values = torch.cat([values, last_value.unsqueeze(0)])  # batch -> batch+1
            advantages = torch.zeros_like(rewards)
            last_advantage = 0
            if not MC:
                for t in reversed(range(len(rewards))):
                    delta = rewards[t] + gamma * values[t + 1] * masks_1[t] - values[t]
                    advantages[t] = last_advantage = delta + gamma * lam * masks_2[t] * last_advantage

                returns = advantages + values[:-1]
            else:

                future_return = 0
                returns = torch.zeros_like(rewards)
                
                for t in reversed(range(len(rewards))):
                    # 计算 TD(λ) 的优势函数
                    delta = rewards[t] + gamma * values[t + 1] * masks_1[t] - values[t]
                    advantages[t] = last_advantage = delta + gamma * lam * masks_2[t] * last_advantage

                    # 计算 Monte-Carlo 的 returns
                    future_return = rewards[t] + gamma * future_return * masks_2[t]
                    returns[t] = future_return

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            del rewards, masks_1, masks_2, values
        
        return advantages, returns # [batch,1] ,[batch,1]
    
    def PPO_learn(self,batch,last_value=None):
        self.iter += 1
        batch_states = batch['states'] # [batch, s_dim]
        batch_actions = batch['actions'] # [batch, a_dim]
        rewards = batch['rewards']  # [batch, ]

        with torch.no_grad():

            # 计算价值
            value = self.value(batch_states).squeeze(-1)
            # 计算旧策略 log_prob
            old_dist = self.pi(batch_states, return_dist=True)
            # 区分离散/连续 计算旧log_prob
            if self.config.get('discrete', False):
                # batch_actions 存的是 int 或 1D 张量，转换为 long
                acts = batch_actions.view(-1).long()
                p_log_probs = old_dist.log_prob(acts).unsqueeze(-1)  # [batch,1]
                #print(p_log_probs.shape)
            else:
                p_log_probs = old_dist.log_prob(batch_actions).sum(dim=-1, keepdim=True)

        # 计算优势函数
        ## mask 处理
        mask_1 = torch.ones_like(rewards) # 最好是使用mask_1 来处理 terminate
        mask_2 = batch['masks_2'] # [batch, ]

        advantages, returns = self.GAE(rewards, value, mask_1, mask_2, gamma=self.config['PPO']['gamma'], lam=self.config['PPO']['lam'],last_value=last_value)        
        advantages = advantages.reshape(-1,1)
        returns = returns.reshape(-1,1)
        

        ppo_mini_batch_size = self.config['PPO']['mini_batch_size']

        for k in range(self.config['PPO']['K_epochs']):
            # 随机打乱数据顺序,
            new_index = torch.randperm(batch_states.shape[0])
            indexs = [new_index[i : i + ppo_mini_batch_size] for i in range(0, batch_states.shape[0], ppo_mini_batch_size)]
            for index in indexs:
                mini_batch_states = batch_states[index]
                mini_batch_actions = batch_actions[index]
                mini_log_probs = p_log_probs[index]   ### 
                mini_advantages = advantages[index]
                mini_returns = returns[index]

                # 计算新的log概率
                p_dist = self.pi(mini_batch_states, return_dist=True)
                if self.config.get('discrete', False):
                    acts_m = mini_batch_actions.reshape(-1).long()  # [batch,1] -> [batch,]
                    new_log_probs = p_dist.log_prob(acts_m).unsqueeze(-1)
                    dist_entropy = p_dist.entropy().unsqueeze(-1).mean()  # entropy per-sample -> mean
                else:
                    new_log_probs = p_dist.log_prob(mini_batch_actions).sum(dim=-1, keepdim=True)
                    dist_entropy = (p_dist.entropy().sum(dim=-1)).mean()

                # 计算比例
                ratio = torch.exp(new_log_probs - mini_log_probs)

                # 计算策略损失
                surr1 = ratio * mini_advantages
                eps = self.config['PPO']['clip_epsilon']
                surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * mini_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.config['PPO']['ent_weight'] * dist_entropy

                # 计算价值损失
                value_pred = self.value(mini_batch_states) # [batch,1]
                value_loss = F.mse_loss(value_pred, mini_returns)

                value_loss = self.config['PPO']['value_loss_coef'] * value_loss

                loss = policy_loss + value_loss

                # 更新所有网络
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.P.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.V.parameters(), 0.5)
                self.optim.step()

    
    def _init_logging(self ,resume: bool = False, start_iteration: int =0):
        '''
        resume: 是否是续训。如果是 True，则复用现有的 log_dir。
        '''
        if not resume:
            # 新训练：创建新目录
            current_time = time.strftime("%Y%m%d-%H%M%S")
            ## 当前文件地址的父文件夹
            current_dir = os.path.dirname(os.path.abspath(__file__)) + '/' 
            self.config["log_dir"] = current_dir + self.config["log_dir"] + '/' + self.config['algo'] + '/' + current_time
            self.config["log_dir"] = os.path.abspath(self.config["log_dir"])
        else:
            # 续训：log_dir 已经在 load_model 里被设置成了 checkpoint 所在的目录
            print(f"Resuming logging in existing directory: {self.config['log_dir']}")
        
        os.makedirs(self.config["log_dir"] , exist_ok=True)
        # purge_step: 如果 TensorBoard 发现步数冲突，会清除该步数之后的记录，防止曲线乱跳（均会清除start_iteration之后的内容）
        self.writer = SummaryWriter(log_dir=self.config["log_dir"] ,purge_step=start_iteration) 
        
        print("Logging to:", self.config["log_dir"])
        

        self.max_reward = None # 修改：初始化为 None，用于判断是否是第一次评估 #-float('inf')
        self.max_ele = -float('inf')
        self.eval_history = []

        # 新增：量化早停方案监控变量（监控 OOS 夏普）
        self.patience_counter = 0
        self.max_patience = self.config['PPO'].get('early_stop_patience', 200) # 默认连续200次无提升则早停

        # 保存配置文件
        self.save_config()
    
    def save_config(self, path: str= None):
        path = path if path is not None else f"{self.config['log_dir']}/config.json"
        import json
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4)

    
    def explore_env(self, env_wrapper):

        #while episode_num < max_episodes:
        while True:
            self.step += 1

            # 与环境交互一步
            state_tensor = torch.as_tensor(self.state, dtype=torch.float32)
            # 这里策略输出对离散/连续两类都兼容：
            action = self.pi(state_tensor.unsqueeze(0))
            # action 可能是 LongTensor (discrete) 或 FloatTensor (continuous)
            if self.config.get('discrete', False):
                # 保证形状为 [action_dim] 或 [1]，保存为 shape [1,1] 以便后续batch一致
                if isinstance(action, torch.Tensor):
                    action_to_env = action.item() if action.dim() == 0 else action.squeeze(0).item()
                    action_save = torch.tensor([[int(action_to_env)]], dtype=torch.long)
                else:
                    action_to_env = int(action)
                    action_save = torch.tensor([[action_to_env]], dtype=torch.long)
            else:
                action = action.squeeze(0).detach()
                action_to_env = action
                action_save = action

            next_state, reward, done, info = env_wrapper.step(self.state, action_to_env)


            # mask: 1 - done，用于 GAE 中的 masks_2
            mask = 1.0 - done
            self.memory.push(state_tensor, action_save, reward, mask)

            self.state = next_state
            self.episode_reward += reward

            # episode 结束：统计一下回报，重新 reset
            if done:
                self.writer.add_scalar('Episode Reward', self.episode_reward, self.episode_num)
                self.train_return.append(self.episode_reward)
                self.new_episode_flag = True  # 新增：标记新回合开始
                self.episode_num += 1
                print(f"Episode {self.episode_num}/{self.max_episodes}, Reward: {self.episode_reward}")
                
                if (self.episode_num) >= self.max_episodes:
                    return None
                self.state = env_wrapper.reset()
                self.episode_reward = 0.0
            
            if self.step % self.horizon == 0:
                mem = self.memory.sample()
                batch = {
                    'states': torch.stack(mem.state).to(self.device),
                    'actions': torch.stack(mem.action).to(self.device),
                    'rewards': torch.stack(mem.reward).to(self.device),
                    'masks_2': torch.stack(mem.mask).to(self.device),
                }
                return batch

    def train_init(self, env_wrapper, num_episodes, model_path: str = None):
        """训练前的初始化工作"""
        if model_path is not None and os.path.exists(model_path):
            start_iter = self.load_model(model_path)
            resume = True
        else:
            resume = False
            start_iter = 0

        self._init_logging(resume=resume, start_iteration=start_iter)

        self.horizon = self.config['PPO']['horizon']
        self.max_episodes = num_episodes

        self.episode_num = start_iter
        self.step = 0
        self.episode_reward = 0.0
        self.train_return = []
        self.iter = 0
        self.new_episode_flag = False  # 新增：标记是否刚开始新回合
        
        self.state = env_wrapper.reset()
        self.memory = Memory(fields=('state', 'action', 'reward', 'mask'))

    def eval_and_save(self, train_env, eval_env):
        """在训练循环中评估和保存模型"""
        should_stop = False  # 新增：是否触发早停
        # 只有当新回合开始时才进行评估
        if not self.new_episode_flag:
            return
        
        if self.episode_num % self.config['PPO']['eval_interval'] == 0 and self.step > 25 * self.horizon :
            self.new_episode_flag = False  # 重置标志位，避免重复评估

            # === 第一次评估：在 train_env 上评估 ===
            avg_reward_step, info = self.evaluate(train_env, self.config['PPO']['eval_episodes'])
            self.writer.add_scalar('Eval/train_env_reward', info['mean_reward'], self.episode_num) # trian_env 总回测的平均回报
            self.writer.add_scalar('Eval/train_env_sharp', info['sharp'], self.episode_num) # trian_env 总回测的回报率
            
            current_reward = info['mean_reward']
            is_best = False
            should_eval_on_eval_env = False
            
            # 判断逻辑
            if self.max_reward is None:
                # 第一次评估，设为最佳
                self.max_reward = current_reward
                is_best = True
                should_eval_on_eval_env = True
                print(f"First evaluation, setting max_reward to: {self.max_reward:.4f}")
            elif current_reward > self.max_reward:
                # 新的最佳模型
                is_best = True
                should_eval_on_eval_env = True
            else:
                # 检查是否接近最佳（相差不到10%）
                if self.max_reward != 0:
                    diff_ratio = abs(current_reward - self.max_reward) / abs(self.max_reward)
                else:
                    diff_ratio = float('inf') if current_reward != 0 else 0
                
                threshold = self.config['PPO'].get('near_best_threshold', 0.1) 
                if diff_ratio < threshold:  # 相差不到10%
                    should_eval_on_eval_env = True
                    print(f"Close to best ({diff_ratio*100:.1f}% diff), will evaluate on eval_env")
            
            # 如果是最佳模型，更新 max_reward 并保存 best 模型
            if is_best:
                self.max_reward = current_reward
                self.save_model(name='best', iteration=self.episode_num)
                self.writer.add_scalar('Eval/train_env_best_reward', self.max_reward, self.episode_num) # # trian_env 总回测的最大回报
                self.writer.add_scalar('Eval/train_env_best_sharp', info['sharp'], self.episode_num) # trian_env 总回测的回报率
                print(f"New best model saved with average return: {self.max_reward:.4f}")
            
            # === 第二次评估：在 eval_env 上评估（最佳或接近最佳时） ===
            if should_eval_on_eval_env:
                avg_reward_step, eval_info = self.evaluate(eval_env, self.config['PPO']['eval_episodes'])
                self.writer.add_scalar('Eval/eval_env_reward', eval_info['mean_reward'], self.episode_num)
                self.writer.add_scalar('Eval/eval_env_sharp', eval_info['sharp'], self.episode_num)
                
                # 无论是否是 train_env 的最佳，只要 eval_env 的 sharp 更好就保存 best_sharp
                if eval_info['sharp'] > self.max_ele:
                    self.max_ele = eval_info['sharp']
                    self.writer.add_scalar('Eval/eval_env_best_sharp', self.max_ele, self.episode_num)
                    self.save_model(name='best_sharp', iteration=self.episode_num,for_pkl=True)
                    self.patience_counter = 0  # 验证集夏普创新高，重置早停计数器
                    print(f"New best_sharp model saved with sharp: {self.max_ele:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"OOS Sharp not improving. Patience: {self.patience_counter}/{self.max_patience}")
                    if self.patience_counter >= self.max_patience:
                        print(f"🚨 触发量化早停！连续 {self.max_patience} 次验证集(OOS)夏普未创新高，防止严重过拟合。")
                        should_stop = True

        return should_stop



    def train_post(self, env_wrapper):
        """训练结束后的收尾工作"""
        self.save_model(name='final', iteration=self.max_episodes) 
        self.evaluate(env_wrapper, 1, for_best_model_eval=True,for_eval_data=self.config['collect_eval_data'])

    
    def train(self, env_wrapper, eval_env_wrapper,num_episodes, model_path: str = None):
        """PPO 独立训练的完整流程"""
        import copy

        # ## 正常是如下：
        self.train_env = env_wrapper
        self.eval_env = eval_env_wrapper  
        self.train_env_ = copy.deepcopy(env_wrapper) # 用于 eval_and_save
        self.train_env_.random_start = False # eval_and_save 中的 train_env_ 不使用随机起始，保持状态连续性

        self.train_init(self.train_env, num_episodes, model_path)

        while self.episode_num < self.max_episodes:
            batch = self.explore_env(self.train_env)
            if batch is None:
                break

            last_value = self.value(torch.as_tensor(self.state, dtype=torch.float32).unsqueeze(0)).squeeze().to(self.device)
            self.PPO_learn(batch, last_value)
            
            # 获取早停标志位
            if self.eval_and_save(self.train_env_,self.eval_env): # 如果还是 self.train_env的话会把self.state 改变
                break

            self.memory.clear()
            
        self.train_post(self.eval_env)
                
    def evaluate(self, env_wrapper, num_episodes: int,new_log:str=None,for_best_model_eval=False,best_model_path=None,for_eval_data=False,fixed_seed_sequence=False) -> float:
        '''
        当每个 Episode 的长度（Step数）是固定的时候。
        avg_reward_step 与 mean_reward 是等价的。
        不固定的时候, avg_reward_step 更能反映每一步的收益情况。
        但是当不固定的时候，奖励设计应该更合理一些，比如加上时间惩罚等，否则智能体会倾向于延长结束 episode。

        '''

        
        if for_best_model_eval:
            best_model_path = best_model_path if best_model_path is not None else os.path.join(self.config["log_dir"] , 'model_best.pth')
            self.load_model(best_model_path)
            env_wrapper.reproducible = False  # 评估最好模型时不固定随机种子
        
        if for_eval_data:
            states = []
            actions = []
            env_wrapper.reproducible = False 
        
        total_reward = []
        total_step = []
        for episode in range(num_episodes):

            if fixed_seed_sequence: # # --- 新增功能：按顺序设置 seed 为 0 ~ num_episodes-1 ---
                state = env_wrapper.reset(seed=episode)
            else:
                state = env_wrapper.reset()
            #state = env_wrapper.reset()
            done = False
            episode_reward = 0.0
            episode_steps = 0
            
            #for _ in range(100):
            while not done:
            #for _ in range(1):

                state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                action = self.pi(state_tensor,return_mean=True).squeeze(0).detach()
                if for_eval_data:
                    states.append(state.cpu().numpy())  
                    actions.append(action.cpu().numpy())
                next_state, reward, done, info = env_wrapper.step(state, action)
                #print(f'{env_wrapper.env.current_date_str}',env_wrapper.env.action,env_wrapper.env.positions,env_wrapper.env.weights),#dict(env_wrapper.env.current_open_prices),
                    #   env_wrapper.env.scores,,# weights
                    #   )
                episode_reward += reward
                episode_steps += 1
                state = next_state
                if done:
                    print('portfolio_value ',env_wrapper.env.portfolio_value)
                    print('sharp',env_wrapper.env.sharp)

                
            total_reward.append(episode_reward.cpu().numpy())
            total_step.append(episode_steps)
        avg_reward_step = [reward/step for reward, step in zip(total_reward, total_step)]
        mean_reward = np.mean(total_reward)
        print('mean reward',mean_reward,total_reward,)

        if new_log is not None:
            self.config["log_dir"] = self.config["log_dir"] + '/' + new_log
            os.makedirs(self.config["log_dir"] , exist_ok=True)
            print("Logging to:", self.config["log_dir"] )
            #print(f"Eval Average step Return: {np.mean(avg_reward_step)}")

        if for_eval_data:
            np.savez(os.path.join(self.config["log_dir"] , 'eval_data.npz'), states=np.array(states), actions=np.array(actions),index = np.array(total_step))
            np.savez(os.path.join(self.config["log_dir"] , 'eval_rewards.npz'), episode_rewards=total_reward, mean_reward=mean_reward)

        # if for_best_model_eval:
        #     #episode_avg_rewards = sum(total_reward) / num_episodes
        #     #np.savez(os.path.join(self.config["log_dir"] , 'eval_rewards.npz'), episode_rewards=total_reward, episode_avg_rewards=episode_avg_rewards)
        #     return episode_avg_rewards , total_reward  
        info={}
        info['mean_reward'] = mean_reward
        info['sharp'] = env_wrapper.env.sharp

        return np.mean(avg_reward_step),info #env_wrapper.env.sharp
    
if __name__ == "__main__":

    '''
    python -m RL.PPO2
    '''


    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv_name = 'train_data_etf_1_ef_ohlc' 
    eval_csv_name = 'trade_data_etf_1_ef_ohlc'
    file_path = os.path.join(current_dir, f'{train_csv_name}.csv')
    eval_file_path = os.path.join(current_dir, f'{eval_csv_name}.csv')


    df = pd.read_csv(file_path)
    df_eval = pd.read_csv(eval_file_path)

    ## 注意 这里df_hs300 得是从同一个csv 里拿出来的（TODO 之后看看可以改成专门一个csv）
    df_hs300 = df[df['tic'] == '000300.SH'][['date', 'close']].reset_index(drop=True)
    df_hs300_eval = df_eval[df_eval['tic'] == '000300.SH'][['date', 'close']].reset_index(drop=True)
    etf_codes = ['518880.SH',
        '159915.SZ', #创业板ETF 成立日：2011-09-20
        '513500.SH', #标普500ETF 成立日：2013-12-05
        '159985.SZ',]


    env = ETFTradingEnv(
        df_price=df, 
        df_benchmark=df_hs300,
        etf_pool=etf_codes,
        m_days=25,
        state_window=20,
        # random_start=True,      # 开启随机起始
        # episode_length=600       # 每个episode固定250天
    ) # 使用所有的天数来训练效果会好一点

    ## seed 环境
    np.random.seed(42) # k可重复 则要将 env_reproducible 为True
    env.action_space.seed(42)

    env_wrapper = gymEnvWrapper_for_env_stock(env, PPO_COBFIG)

    env = ETFTradingEnv(
            df_price=df_eval, 
            df_benchmark=df_hs300_eval,
            etf_pool=etf_codes,
            m_days=25,
            state_window=20,
            random_start=False       # 固定起始
        )
    eval_env_wrapper = gymEnvWrapper_for_env_stock(env, PPO_COBFIG)

    ppo = PPO(PPO_COBFIG)
    train = 0

    if train:
        ppo.train(env_wrapper, eval_env_wrapper, num_episodes=1000)

    eval = 1-train 
    if eval:

        file = os.path.join( current_dir,rf'logs\env_etf\PPO') 
        best_model_path = os.path.join(file,'20260329-233741','model_best_sharp.pth') 

        ppo.evaluate(eval_env_wrapper, num_episodes=1,for_best_model_eval=True,fixed_seed_sequence=True,best_model_path=best_model_path)


        # 画一下环境中的净值曲线
        
        env.render(plot=True, save_path=os.path.join(ppo.config["log_dir"], 'eval_plot.png'))



    ###
    ## 之后尝试 使用sb3 的PPO 来对比一下

    '''
    以夏普比高为最终目标。
    奖励设置为当日收益率。

    1.2020.0309 - 2023.0213 训练 环境1
    2.2023.0214 - 2026.0114 评估 环境2
    保存模型逻辑，
    {
        在环境1上评估，如果是reward最高,或者reward接近最高（相差不到10%）
        就在环境2上评估，如果环境2的sharp更好就保存best_sharp模型。
    }
    训练后环境2 best_sharp为最终模型，且好于在环境2上使用其他所有动作的sharp。（其他动作表示市面上常见策略）
    最后在2026.0114 - 2026.0318 评估环境2的最终模型的sharp 依旧好于其他所有动作的sharp。可以证明模型有效。

    '''

    '''
    评估指标
    第一步：看“性价比”（效率指标）
    —— 核心看：夏普比率、索提诺比率、信息比率在看赚多少钱之前，先看赚这些钱稳不稳。夏普比率 (1.366)：这是你的“及格线”。在 A 股回测中，夏普过 $1$ 算优秀，过 $1.5$ 算顶级。$1.366$ 说明你每承担 $1\%$ 的风险，能换来 $1.36$ 倍的超额收益。
    索提诺比率 (1.919)：这个数字比夏普更高，是极大的利好。它只计算“下行风险”（即跌的时候的波动）。索提诺远高于夏普，说明你的策略波动很多时候是“向上暴涨”带来的，而不是“阴跌”带来的。
    信息比率 (1.482)：这是评价你“超越基准能力”的指标。$1.48$ 非常高，说明你的策略相对于指数的超额收益非常稳定，不是靠某几天运气好。
    第二步：看“生存力”（风险指标）
    —— 核心看：最大回撤、超额收益最大回撤最大回撤 (25.38%)：这是你的策略最难受的时候。分析日期：你给出的时间点非常关键——2024/10/08 到 2024/10/17。背景归因：2024 年 10 月 8 日是 A 股近期那个疯狂涨停开盘、随后大幅剧震的日子。你的策略在这里产生最大回撤，说明它在极端情绪反转、波动率瞬间炸裂的情况下，风控保护可能稍显滞后。
    超额回撤 (20.03%)：虽然总资产跌了 $25\%$，但相对于基准只多跌了 $20\%$。这说明那段时间大盘也在崩，你被带下来了一部分。
    第三步：看“骨架”（交易逻辑）—— 核心看：盈亏比、胜率这两个指标决定了策略的“性格”：盈亏比 (3.003)：这是你这份数据中最惊艳的地方。这意味着你平均每亏 $1$ 块钱的时候，赚的时候能赚 $3$ 块。
    胜率 (59.1%)：在量化策略里，尤其是趋势或 RL 策略，胜率能接近 $60\%$ 配合 $3$ 倍盈亏比，简直是“印钞机”组合。
    结论：你的策略属于 “大赚小赔”型。它不靠频繁的小胜，而是靠抓住几次大的爆发，并且止损做得非常果断。
    第四步：看“归因”（收益来源）—— 核心看：阿尔法 (Alpha)、贝塔 (Beta)
    阿尔法 (0.357)：这代表你靠“真本事”赚的钱。$0.35$ 意味着你独立于大盘走势之外，每年能多跑赢约 $35\%$。
    贝塔 (0.731)：这个值小于 $1$。说明你的策略并不激进，它只承担了市场 $73\%$ 的波动。
    低 Beta、高 Alpha 是所有量化基金经理终极追求的“圣杯”状态。
    '''

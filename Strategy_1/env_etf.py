'''
环境参考：
# 克隆自聚宽文章：https://www.joinquant.com/post/63252
# 标题：DQN-ETF强化学习-滚动-经验保存反复学习版
# 作者：lq123456789

# 克隆自聚宽文章：https://www.joinquant.com/post/62904
# 标题：AI强化学习ETF策略-DQN智能体-滚动训练
# 作者：神启


为了之后可以拿训练好的模型放在聚宽上回测： 指标计算 在环境里写成函数形式封装。


'''
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import math


##
'''
动作先按这个来
'''

class ETFTradingEnv(gym.Env):
    """
    ETF 轮动策略环境 (参考聚宽 DQN-ETF 策略)
    
    动作空间 (Discrete 10):
        0-2: 等权重 (持有 Top 1, 2, 3)
        3-5: 动量权重 (持有 Top 1, 2, 3)
        6-8: 波动率权重 (持有 Top 1, 2, 3)
        9: 空仓 (Cash)
    
    状态空间 (Box):
        [市场波动率, 市场动量, 平均ETF动量, 动量离散度, 相关性代理, ETF1动量, ETF2动量, ..., ETFn动量]
        维度 = 5 + ETF数量
    
    注意：此环境为 etf 买卖交易为万2佣金，不考虑滑点，不考虑手续费最小值。
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 df_price: pd.DataFrame,   # 包含所有ETF价格的长格式数据 [date, tic, close, ...]
                 df_benchmark: pd.DataFrame, # 基准指数数据(如沪深300) [date, close]
                 etf_pool: list,           # ETF 代码列表
                 m_days: int = 25,         # 动量计算窗口
                 volatility_days: int = 60,    # 波动率计算窗口
                 state_window: int = 20,   # 市场状态计算窗口
                 initial_amount: int = 10000,
                 transaction_cost_pct: float = 0.0002, # 万2佣金
                 reward_scaling: float = 1e-4,
                # === 随机化参数 ===
                 random_start: bool = False,       # 是否随机起始
                 episode_length: int = 0,        # 每个episode固定长度（交易日）
                 ):
        
        self.df_price = df_price.sort_values(['date', 'tic']).reset_index(drop=True)
        self.df_benchmark = df_benchmark.sort_values('date').reset_index(drop=True)
        self.etf_pool = etf_pool
        self.n_etfs = len(etf_pool)
        
        self.m_days = m_days
        self.volatility_days = volatility_days
        self.state_window = state_window
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling

        # 随机化参数
        self.random_start = random_start
        self.episode_length = episode_length
        
        # 定义动作空间 (参照聚宽策略)
        # 0-8: 不同的持仓组合, 9: 空仓
        # 内部映射：动作ID -> 配置
        self.action_map = [
            {'type': 'equal', 'num': 1}, {'type': 'equal', 'num': 2}, {'type': 'equal', 'num': 3},
            ##{'type': 'momentum', 'num': 1}, 
            {'type': 'momentum', 'num': 2}, {'type': 'momentum', 'num': 3},
            ##{'type': 'volatility', 'num': 1}, 
            {'type': 'volatility', 'num': 2}, {'type': 'volatility', 'num': 3},
            {'type': 'cash', 'num': 0}
        ]
        self.action_num = len(self.action_map)
        self.action_space = spaces.Discrete(self.action_num)
        
        # 定义状态空间 (5个市场特征 + N个ETF动量分)
        self.state_dim = 5 + self.n_etfs
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        
        # 预处理数据：将数据转换为以日期为索引，列为tic的宽表，方便快速查找
        self.price_matrix = self.df_price.pivot(index='date', columns='tic', values='close')
        # 确保列序与 etf_pool 一致
        self.price_matrix = self.price_matrix[self.etf_pool] 
        self.dates = self.price_matrix.index.tolist()

        # 新增: 预处理开盘价矩阵，用于交易执行
        self.open_matrix = self.df_price.pivot(index='date', columns='tic', values='open')
        self.open_matrix = self.open_matrix[self.etf_pool]
        
        # 对齐基准数据
        self.benchmark_series = self.df_benchmark.set_index('date')['close']
        self.benchmark_series = self.benchmark_series.reindex(self.dates).ffill() #self.benchmark_series.reindex(self.dates).fillna(method='ffill')


        # 定义需要跳过的问题日期列表
        self.BAD_DATES = ['2021-02-08', '2022-03-29']

        # 计算可用的起始范围         # 因为计算动量和状态需要历史数据，所以起始日期不能是第0天
        self.min_start_day = max(self.m_days, self.state_window, self.volatility_days)
        # 最大起始日：需要留出 episode_length 天 + 1（最后一天不交易）
        self.max_start_day = len(self.dates) - self.episode_length - 1
        
        if self.max_start_day <= self.min_start_day:
            print(f"[WARNING] 数据不足以支持随机起始: 总天数={len(self.dates)}, "
                  f"min_start={self.min_start_day}, 需要episode_length={self.episode_length}")
            self.max_start_day = self.min_start_day  # 退化为固定起始
        
        self.reset()
        ## 相当于每天开盘前做决策，收盘后结算
        self.strat_day = self.day
        self.terminal_day = self.day + self.episode_length if self.random_start else len(self.dates) - 2
        print(f"环境初始化完成，数据日期范围：{self.dates[self.strat_day]} 至 {self.dates[min(self.terminal_day, len(self.dates)-1)]}，"
              f"总交易日数：{self.terminal_day - self.strat_day + 1} 天。"
              f"{'（随机起始模式）' if self.random_start else '（固定起始模式）'}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        

        # 合并 options 中的设置（允许外部临时覆盖）
        random_start = self.random_start
        if options is not None:
            random_start = options.get('random_start', random_start)
        
        if random_start and self.max_start_day > self.min_start_day:
            # 随机选择起始日
            self.day = self.np_random.integers(self.min_start_day, self.max_start_day + 1) # intergers 是左闭右开，所以 max_start_day + 1
            self.terminal_day = self.day + self.episode_length
        else:
            # 固定起始：从最早可用日期开始，走到数据末尾
            self.day = self.min_start_day
            self.terminal_day = len(self.dates) - 2
        
        if self.episode_length:
            self.terminal_day = self.day + self.episode_length
        else:
            self.terminal_day = len(self.dates) - 2

        
        self.start_day = self.day
        
        self.portfolio_value = self.initial_amount
        self.cash = self.initial_amount
        self.positions = {tic: 0 for tic in self.etf_pool} # 持仓股数
        self.last_portfolio_value = self.initial_amount
        
        # 记录历史
        self.portfolio_value_history = [self.initial_amount]
        
        self.data_memory = []
        self.current_rate = 0
        self.return_rate_memory = []

        self.step_return_memory = []

        # 初始化状态
        self.state = self._get_state(self.day)
        
        return self.state, {}

    def step(self, action):
        self.action = action
        

        # --- 简化修改 1: 直接跳过特定坏日期 ---
        current_date = self.dates[self.day]
        
        # 将日期转换为字符串比较，或者根据你的数据格式调整
        self.current_date_str = str(current_date) if isinstance(current_date, str) else current_date.strftime('%Y-%m-%d')
        
        
        if self.current_date_str in self.BAD_DATES:
            self.day += 1

        # 1. 获取当前日期(T日)用于决策，实际交易在T日收盘(简化)或T+1开盘
        # 这里模拟：根据T-1日及之前的数据计算信号，在T日按收盘价成交
        current_date = self.dates[self.day]
        current_close_prices = self.price_matrix.iloc[self.day]
        current_open_prices = self.open_matrix.iloc[self.day] # 当前日 开盘价 调仓
        self.current_open_prices = {tic: current_open_prices[tic].item() for tic in self.etf_pool}
        

        # 2. 执行交易逻辑
        action_config = self.action_map[action]
        weights = self._calculate_allocation(action_config, self.day)
        self.weights = {tic: float(weights[tic]) for tic in weights} # 记录当前权重分配


        
        # 3. 计算交易后的资产组合
        # 先以当前价格计算调仓前的总资产 : 即当前的total_value
        pre_trade_value = self.cash + sum([self.positions[tic] * current_open_prices[tic] for tic in self.etf_pool if not np.isnan(current_close_prices[tic])])
        
        # 简化的调仓：假设全部卖出再买入 (考虑交易成本)
        # 实际总资产 = 调仓前价值 * (1 - 换手造成的成本)
        # 这里为了计算简便，计算目标持仓价值，并扣除变动部分的交易成本
        
        new_positions = {tic: 0 for tic in self.etf_pool}
        transaction_cost = 0
        total_turnover_value = 0  # 新增：累计换手金额
        
        # 目标每个ETF的资金分配
        target_values = {tic: pre_trade_value * w for tic, w in weights.items()}
        
        ## 这里买卖实盘得变成，先卖后买
        for tic in self.etf_pool: 
            price = current_open_prices[tic]
            if np.isnan(price) or price <= 0: 
                ## 异常值处理2 ## TODO 不对：万一是调这个仓呢
                #new_positions[tic] = self.positions[tic] # 继承原有持仓
                continue
            
            # 目标股数 - 修改：强制为100的倍数 (1手=100股)
            #target_shares = int(target_values.get(tic, 0) / price) 
            raw_shares = int(target_values.get(tic, 0) / price)
            target_shares = (raw_shares // 100) * 100
            
            # 变动股数
            diff_shares = abs(target_shares - self.positions[tic])
            
            # 交易成本 = 变动金额 * 费率
            change_value = diff_shares * price
            transaction_cost += change_value * self.transaction_cost_pct
            total_turnover_value += change_value
            
            new_positions[tic] = target_shares
            
        # 更新持仓和现金
        self.positions = new_positions
        
        
        # 计算交易完成后的现金剩余: 
        # 现金 = 调仓前总资产(按Open计) - 新持仓市值(按Open计) - 交易成本
        current_holdings_cost_at_open = sum([self.positions[tic] * current_open_prices[tic] for tic in self.etf_pool if not np.isnan(current_open_prices[tic])])
        self.cash = pre_trade_value - current_holdings_cost_at_open - transaction_cost
        
        # 新增：保存换手金额供奖励计算使用
        self.last_turnover_value = total_turnover_value
        # 4. 计算当日收盘总净值 (Portfolio Value)
        # 用 Close 价格计算持仓市值 + 现金
        current_holdings_value_at_close = sum([self.positions[tic] * current_close_prices[tic] for tic in self.etf_pool if not np.isnan(current_close_prices[tic])])
        self.portfolio_value = self.cash + current_holdings_value_at_close

        # print('positions after trade:', self.positions)
        # print('cash after trade:', self.cash)

        
        # 4. 前进一步
        self.day += 1
        # === 修改：使用动态的 terminal_day 判断终止 ===
        self.terminal = self.day >= self.terminal_day
        
        # 5. 更新状态
        next_state = self._get_state(self.day)
        
        # 6. 计算奖励
        # 奖励公式参考聚宽策略：portfolio_return * 252 - risk_penalty (这里简化为单步收益)
        step_return = (self.portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
        self.step_return_memory.append(step_return)
        
        reward = step_return
            
        self.last_portfolio_value = self.portfolio_value
        
        # 记录
        self.state = next_state
        self.portfolio_value_history.append(self.portfolio_value)

        self.data_memory.append(self.current_date_str)
        self.current_rate = (self.portfolio_value / self.initial_amount) - 1 #step_return
        self.return_rate_memory.append(self.current_rate)

        # 新增：在 episode 结束时计算并记录夏普比率
        if self.terminal:
            self.sharp = self._calculate_sharpe_ratio()
        
        return self.state, reward, self.terminal, False, {}
    

    def _get_valid_history(self, tic, day_idx, m_days):
        """
        获取截止到 day_idx 的 m_days 个有效数据点（剔除 <=0 和 NaN）。
        采用增量回补策略，效率更高。

        此函数为解决坏日期
        """
        # 1. 初始尝试：获取理论上的 m_days 数据
        start_idx = max(0, day_idx - m_days)
        raw_series = self.price_matrix[tic].iloc[start_idx:day_idx]
        
        # 2. 筛选有效数据
        valid_mask = (raw_series > 1e-4) & (raw_series.notna())
        valid_series = raw_series[valid_mask]
        
        # 3. 检查缺失数量
        missing_count = m_days - len(valid_series)
        
        # 4. 如果有缺失，往前多拿 "missing_count" 个数据（并考虑可能往前拿的数据里也有坏数据，适当多拿一点）
        if missing_count > 0 and start_idx > 0:
            # 简单策略：缺多少补多少，再乘个系数防止补的那段也有坏账，通常 1.5倍足够
            # 比如缺 1 个，我往前多拿 2 天；缺 5 个，多拿 8 天
            lookback_buffer = int(missing_count * 1.5) + 1  
            new_start_idx = max(0, start_idx - lookback_buffer)
            
            # 重新获取更长的一段
            raw_series = self.price_matrix[tic].iloc[new_start_idx:day_idx]
            valid_mask = (raw_series > 1e-4) & (raw_series.notna())
            valid_series = raw_series[valid_mask]
        
            # 如果还是不够（极个别极端停牌情况），再兜底拿全量，或者直接接受现状
            if len(valid_series) < m_days:
                 raw_series = self.price_matrix[tic].iloc[:day_idx]
                 valid_series = raw_series[(raw_series > 1e-4) & (raw_series.notna())]

        # 5. 截取最后 m_days 个
        return valid_series.iloc[-m_days:] if len(valid_series) >= m_days else pd.Series(dtype=float)
    
    def _get_state(self, day_idx):
        """参考 get_market_state 构建状态"""
        # 1. 计算动量得分 (基于 day_idx 之前的数据)
        # 获取用于计算动量的历史数据窗口 [day_idx - m_days : day_idx]
        # 注意：pandas切片是左闭右开，且我们要算的是截至到 day_idx (不含未来)
        
        # 动量计算窗口数据
        # hist_start = day_idx - self.m_days
        # if hist_start < 0: hist_start = 0
        # hist_prices = self.price_matrix.iloc[hist_start:day_idx]

        
        self.momentum_scores = {} 
        for tic in self.etf_pool:
            #series = hist_prices[tic].dropna() # 不会过滤0 
            # --- 修改：调用封装好的增量获取函数 ---
            series = self._get_valid_history(tic, day_idx, self.m_days)
            if len(series) < self.m_days:
                self.momentum_scores[tic] = 0
                continue
            
            # 计算简单的年化收益率作为动量 (参照 get_rank 中的 slope)
            # y = log(close), x = index
            try:
                y = np.log(series.values)
                x = np.arange(len(y))
                slope, intercept = np.polyfit(x, y, 1)
                
                # 计算 R^2
                y_pred = slope * x + intercept
                #ss_res = sum((y - y_pred) ** 2)
                #ss_tot = np.sum((y - np.mean(y)) ** 2)
                #r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                r_squared = 1 - (sum((y - y_pred)**2) / ((len(y) - 1) * np.var(y, ddof=1)))
                
                annualized_returns = math.pow(math.exp(slope), 250) - 1
                score = annualized_returns * r_squared
                self.momentum_scores[tic] = score
            except:
                self.momentum_scores[tic] = 0 # 一般不会触发 

        # 2. 构建市场特征
        # 市场波动率
        bench_start = day_idx - self.state_window
        bench_data = self.benchmark_series.iloc[bench_start:day_idx]
        
        # bench_ret = bench_data.pct_change().dropna()
        # market_volatility = bench_ret.std() * np.sqrt(252) if not bench_ret.empty else 0
        
        # 使用下面这种计算方法 市场波动率 
        bench_data_ = pd.concat([bench_data.iloc[[0]], bench_data])
        bench_array = np.array(bench_data_)
        bench_ret = np.diff(bench_array) / bench_array[:-1]  # (P_t - P_{t-1}) / P_{t-1}
        market_volatility = np.std(bench_ret, ddof=1) * np.sqrt(250) 

        
        # 市场累计年化收益  ## 这里好像不一样 改成如下
        #bench_day_num = len(bench_data)
        #bench_ratio = (bench_data.iloc[-1] - bench_data.iloc[0]) / bench_data.iloc[0]
        ##market_momentum = (1 + bench_ratio) ** (250 / bench_day_num) - 1 #市场累计年化收益计算方式
        
        # 市场动量 使用原版下述方式好像效果更好。: 简单的线性年化（不是复利），在量化策略里非常常见，用来做市场择时（趋势跟踪）
        market_momentum = (bench_data.iloc[-1] / bench_data.iloc[0] - 1) * 252 / len(bench_data) if len(bench_data) > 0 else 0
        
        # ETF池统计特征
        scores_list = list(self.momentum_scores.values())
        avg_etf_momentum = np.mean(scores_list)
        momentum_dispersion = np.std(scores_list)
        
        # 相关性代理 (Top 2 差异)
        sorted_scores = sorted(scores_list, reverse=True)
        correlation_proxy = abs(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 0
        
        market_features = np.array([
            market_volatility, market_momentum, avg_etf_momentum, 
            momentum_dispersion, correlation_proxy
        ])
        
        # 归一化处理 (简单标准化) 对于量化来说：截面因子归一
        if np.std(market_features) > 0:
            market_features = (market_features - np.mean(market_features)) / (np.std(market_features) + 1e-8)
            
        # ETF特征
        etf_features = np.array([self.momentum_scores[tic] for tic in self.etf_pool])
        if np.std(etf_features) > 0:
            etf_features = (etf_features - np.mean(etf_features)) / (np.std(etf_features) + 1e-8)
            
        full_state = np.concatenate([market_features, etf_features])

        return np.nan_to_num(full_state)

    def _calculate_allocation(self, action_config, day_idx):
        """根据动作配置计算权重"""
        action_type = action_config['type']
        num_target = action_config['num']
        
        if action_type == 'cash' or num_target == 0:
            return {}
            
        # 使用 _get_state 计算好的动量分数来排序选股
        self.scores = scores = self.momentum_scores
        # 排序 注意：只对动量分数 > 0 的标的进行排序，确保选出的标的具有正动量
        ranked_etfs = sorted([k for k, v in scores.items() if v > 0], key=lambda x: scores[x], reverse=True)
        
        if not ranked_etfs:
            return {} # 无正动量标的，空仓
            
        target_etfs = ranked_etfs[:min(num_target, len(ranked_etfs))]
        
        weights = {}
        if action_type == 'equal':
            for tic in target_etfs:
                weights[tic] = 1.0 / len(target_etfs)
                
        elif action_type == 'momentum':
            # 参考 calculate_weights momentum 逻辑
            target_scores = [scores[tic] for tic in target_etfs]
            min_score = min(target_scores)
            # 由于已排除动量<0 的, 这里最小为0.01，避免负数和0权重 合理 
            adj_scores = [max(s - min_score + 0.01, 0.01) for s in target_scores]
            total_score = sum(adj_scores)
            for i, tic in enumerate(target_etfs):
                weights[tic] = adj_scores[i] / total_score
                
        elif action_type == 'volatility':
            # 计算波动率倒数权重
            inv_vols = []
            valid_targets = []
            for tic in target_etfs:
                # 取60天波动率
                prices = self._get_valid_history(tic, day_idx, self.volatility_days)
                ret = prices.pct_change().dropna()
                vol = ret.std() if not ret.empty else 1
                inv_vols.append(1.0 / vol)
                valid_targets.append(tic)                
            
            total_inv_vol = sum(inv_vols)
            for i, tic in enumerate(valid_targets):
                weights[tic] = inv_vols[i] / total_inv_vol
                
        return weights

    def _calculate_sharpe_ratio(self):
        """
        计算当前的年化夏普比率
        """
        day_num = len(self.return_rate_memory)
        if day_num <= 1:
            return 0.0
            
        pv_array = np.array(self.portfolio_value_history)
        # 用绝对净值计算每日真实收益率
        daily_rets = np.diff(pv_array) / pv_array[:-1]  
        
        # 计算年化收益率
        R_p = (1 + self.current_rate) ** (250 / day_num) - 1
        R_f = 0.04 # 无风险利率设定为4%
        
        # 计算组合年化波动率（样本标准差 ddof=1）
        sigma_p = np.std(daily_rets, ddof=1) * np.sqrt(250)
        
        sharpe_ratio = (R_p - R_f) / sigma_p if sigma_p > 0 else 0.0
        return sharpe_ratio
    
    
    def _calculate_AEI(self):
        '''
        计算日均超额收益率（Average Excess Return）
        AEI = (1/N) * Σ(EI_i - EI_i-1)
        EI_i = (策略收益+1) / (基准收益+1) - 1
        '''
        N = len(self.return_rate_memory)
        if N <= 1:
            return 0.0
            
        # 1. 确定基准起点序列
        bench_index_strs = [str(d)[:10] if hasattr(d, 'strftime') else str(d)[:10] for d in self.benchmark_series.index]
        bench_dict = dict(zip(bench_index_strs, self.benchmark_series.values))
        
        # 2. 提取有效交易日的基准价格和策略净值
        p_values = self.portfolio_value_history # 包括初始资金在内长度为 N+1
        b_values = []
        
        # 获取 T-1 的基准作为起跑线
        start_idx = max(0, self.start_day - 1)
        b_0 = self.benchmark_series.iloc[start_idx]
        b_values.append(b_0)
        
        for dt_str in self.data_memory:
            dt_key = dt_str[:10]
            # 如果日期缺失，继承上一日基准
            b_val = bench_dict.get(dt_key, b_values[-1])
            b_values.append(b_val)
            
        # 3. 转换为 Numpy 数组方便计算每日单日涨跌幅
        p_array = np.array(p_values)
        b_array = np.array(b_values)
        
        # 单日真实绝对收益 = (今天 / 昨天) - 1
        daily_strat_ret = np.diff(p_array) / p_array[:-1]
        daily_bench_ret = np.diff(b_array) / np.array(b_array[:-1], dtype=float)
        
        # =========================================================
        # 方式 A：标准行业算法 (对应多数云平台的每日对冲平均）
        # AEI = 每日 (策略涨幅 - 基准涨幅) 的平均
        # daily_excess = daily_strat_ret - daily_bench_ret
        # aei_standard = np.mean(daily_excess)

        ## 聚宽除法版 更接近聚宽上的面板
        daily_excess = (daily_strat_ret+1) / (daily_bench_ret+1) -1
        aei_standard = np.mean(daily_excess)
        
        # 方式 B：你原注释的公式代数化简版 (EI_N / N) 与聚宽相差略大
        # EI_N = 最终总策略净值比 / 最终总基准净值比 - 1
        # total_strat_ratio = p_array[-1] / p_array[0]
        # total_bench_ratio = b_array[-1] / b_array[0]
        # ei_N = (total_strat_ratio / total_bench_ratio) - 1
        # aei_cumulative = ei_N / N
        # =========================================================
        
        # 这里返回标准算法，最贴近系统的多日表现追踪
        return float(aei_standard) 


    
    def render(self, mode='human',plot=False,save_path=None):
        print(f"Day: {self.day}, Value: {self.portfolio_value:.2f}")
        
        ## 计算指标

        # ##计算夏普比
        
        ##计算夏普比
        sharpe_ratio = self._calculate_sharpe_ratio()
        print(f"Cumulative Return: {self.current_rate:.2%}, Sharpe Ratio: {sharpe_ratio:.2f}")
        
        ## 计算日均超额收益
        aei = self._calculate_AEI()
        print(f"Average Excess Return (AEI): {aei:.4f}")
                
        # 画一下asset_memory 曲线
        import matplotlib.pyplot as plt
        

        assert len(self.data_memory) == len(self.return_rate_memory)

        # 确保有数据可画
        if plot:
            if len(self.data_memory) > 0 and len(self.return_rate_memory) > 0:
                
                kedu = 20

                plt.figure(figsize=(18, 6))
                
                # 绘制曲线
                plt.plot(self.data_memory, self.return_rate_memory, label='Cumulative Return Rate', color='blue')
                
                plt.title(f'Strategy Return Rate Curve ({len(self.data_memory)} Days, time interval {kedu} Days)')
                plt.xlabel('Date')
                plt.ylabel('Cumulative Return Rate')
                plt.grid(True)           
                plt.legend()
                
                # 优化X轴日期显示：如果数据点太多，只显示部分标签
                total_points = len(self.data_memory)
                if total_points > 15:
                    # 比如只显示10个刻度
                    step = total_points // kedu
                    x_list = list(range(0, total_points, step)) + [total_points - 1]
                    plt.xticks(
                        ticks=x_list, 
                        labels=[self.data_memory[i] for i in x_list],
                        rotation=45
                    )
                else:
                    plt.xticks(rotation=45)
                
                plt.tight_layout()
                
                if save_path is not None:
                    plt.savefig(save_path)
                    print(f"Return rate curve saved to {save_path}")
                plt.show()



if __name__ == "__main__":
    import os
    import numpy as np

    '''
    目前环境为在t 天使用t-1及之前的数据计算信号，在t 天开盘价执行交易，t 天收盘价计算净值。（即：开盘价卖）
    但是 聚宽效果上 一般使用10:30 进行交易比较好 但是对于目前没有10：30信息

    这里是发现使用close 价比open价要好 ；但是如果 在open价训练的策略更好，那么在close价应该还要好（推测）
    '''

    '''
    此数据有两个坑,
    ⚠️ 发现数据不完整的代码 (将被移除):
   ❌ 159915.SZ: 缺失 1 天 (总 1481 天)
      最早缺失: 2021-02-08 00:00:00, 最晚缺失: 2021-02-08 00:00:00 需之后代码处理
   ❌ 513500.SH: 缺失 1 天 (总 1481 天)
      最早缺失: 2022-03-29 00:00:00, 最晚缺失: 2022-03-29 00:00:00 需之后代码处理

    处理：跳过当天
    
    '''

    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv_name = 'train_data_etf_1_ef_ohlc' 
    eval_csv_name = 'trade_data_etf_1_ef_ohlc'
    file_path = os.path.join(current_dir, f'{eval_csv_name}.csv')

    
    df = pd.read_csv(file_path)

    # 2. 确保日期列是 datetime 格式
    # 注意：请将 'date' 替换为你实际的日期列名称
    df['date'] = pd.to_datetime(df['date'])

    # # 3. 设置你想要截取的起始日期 (例如从 2020年1月1日 开始)
    # start_date = '2022-06-30'

    # # 4. 截取该日期及之后的数据
    # df = df[df['date'] >= start_date]

    ## 注意 这里df_hs300 得是从同一个csv 里拿出来的（TODO 之后看看可以改成专门一个csv）
    df_hs300 = df[df['tic'] == '000300.SH'][['date', 'close']].reset_index(drop=True)
    etf_codes = ['518880.SH', # 2013-07-31
        '159915.SZ', #创业板ETF 成立日：2011-09-20
        '513500.SH', #标普500ETF 成立日：2013-12-05
        '159985.SZ',] # 豆粕etf 成立日 2019-12-31
    #df['tic'].unique().tolist()

    env = ETFTradingEnv(
        df_price=df, 
        df_benchmark=df_hs300,
        etf_pool=etf_codes,
        m_days=25,
        state_window=20,
        #episode_length=30,
    )

    ## seed 环境
    np.random.seed(42)
    env.action_space.seed(42)


    # 3. 测试运行
    for i in range(8): # 8
        print(f"\n=== Episode {i} ===")
        step = 0
        obs, _ = env.reset()
        done = False
        #print(obs)
        while not done:
        #for _ in range(100):
            action = i #6 #i #env.action_space.sample() # 随机动作  0 : 24618.65 随机：19287.84
            next_obs, reward, done, _, info = env.step(action)
            step += 1
            if done:
                print(f"{step} : Action: {action}, Reward: {reward:.4f}, Portfolio Value: {env.portfolio_value:.2f}, positions: {env.positions}")
            obs = next_obs
        
        env.render()
    print()



# 万2佣金下

import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将需要复用的值存入变量
run_id = 'env_etf' # 'water-test' ; 'Pendulum-v1-test'

PPO_COBFIG = {
    # 训练id
    'id': run_id,
    'log_dir': f'logs/{run_id}',
    'algo': 'PPO',
    'seed': 42,
    'env_reproducible': True,
    'collect_eval_data': True,

    'num_workers': 1,
    'use_gpu': True,  # 确保启用GPU


    # 's_dim': 9,
    # 'a_dim': 1,

    's_dim': 9,
    'a_dim': 8,

    'discrete': True, # True for discrete action space, False for continuous action space

    
    'PPO': {
        ## liner
        'policy_mlp_dim': 128,
        'activation':'ReLU', # ReLU, Leakyrelu, Elu, Swish
        'dropout':0.01, #0.01,
        'layernorm':False,

        ## actor log_std range
        'log_std_min': -10,
        'log_std_max': 2,

        'p_lr': 1e-3,#5e-4,#1e-3,
        'v_lr': 1e-3,#5e-4,#1e-3,
        'horizon': 2048,#256,#2048,
        'gamma': 0.99,#0.999, #0.99,
        'lam': 0.95,
        'clip_epsilon': 0.2,
        'K_epochs': 10,#3,#10,
        'value_loss_coef': 1,
        'ent_weight': 0.01,#0.1,#0.01,
        'mini_batch_size': 64,#8,#64,
        'eval_interval': 1,
        'eval_episodes':1,
    },

}

'''
activation 可选参数：
[
    "Threshold",
    "ReLU",
    "RReLU",
    "Hardtanh",
    "ReLU6",
    "Sigmoid",
    "Hardsigmoid",
    "Tanh",
    "SiLU",
    "Mish",
    "Hardswish",
    "ELU",
    "CELU",
    "SELU",
    "GLU",
    "GELU",
    "Hardshrink",
    "LeakyReLU",
    "LogSigmoid",
    "Softplus",
    "Softshrink",
    "MultiheadAttention",
    "PReLU",
    "Softsign",
    "Tanhshrink",
    "Softmin",
    "Softmax",
    "Softmax2d",
    "LogSoftmax",
]
'''
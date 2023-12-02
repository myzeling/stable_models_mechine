import torch
import argparse
from exp.exp_main import Exp_Main
import random
import numpy as np
import os
import pandas as pd
import pickle
import feather

parser = argparse.ArgumentParser(description='article methods')

# basic config
parser.add_argument('--model',type = str, default = 'LGB', help = 'model name')
parser.add_argument('--target',type = str, default = 'ret_next', help = 'target name')
parser.add_argument('--split_date',type = str, default = '2020-03-31', help = 'split date')

# data paths
parser.add_argument('--path_factors',type = str, default = 'C:/Users/dcl/Desktop/vscode项目/daily-monthly/异步更新/factors.feather', help = 'factor path')
parser.add_argument('--path_macros',type = str, default = 'C:/Users/dcl/Desktop/vscode项目/calculate factors/Macros/Macro_factor.csv', help = 'macro path')

# formers path
parser.add_argument('--path_formers',type = str, default = './formers', help = 'former path')

# result paths
parser.add_argument('--checkpoints',type = str, default = './results/checkpoints', help = 'checkpoints path')
parser.add_argument('--path_net_value',type = str, default = './results/net_value/net_value.feather', help = 'net value path')
parser.add_argument('--net_value_fig',type = str, default = './results/net_value/figs/net_value.png', help = 'value fig path')
parser.add_argument('--result_path',type = str, default = './results', help = 'Roos result path')
parser.add_argument('--path_importance',type = str, default = './results/importance.csv', help = 'importance path')
parser.add_argument('--path_heatmap',type = str, default = './results/heatmap_importance.png', help = 'heatmap path')
parser.add_argument('--path_importance_scoring',type = str, default = './results/importance_scoring.csv', help = 'importance scoring path')
# model parameters
parser.add_argument('--epsilon',type = float, default = 1.35, help = 'epsilon of OLSH')
parser.add_argument('--formers',type = bool, default = False, help = 'whether to process former data')
# result parameter
parser.add_argument('--per',type = float, default = 0.1, help = 'percentage of buying')
if __name__ == '__main__':
    args = parser.parse_args()
    
    exp = Exp_Main(args)
    if args.formers:
        exp.getformers()
    else:
        exp.train()
    print('done')
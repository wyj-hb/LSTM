# @Time : 2024-05-16 12:44
# @Author : wyj
# @File : parser_my.py
# @Describe : 项目配置
import argparse

import torch

parser = argparse.ArgumentParser(description="项目配置")
# print(parser.description)
# TODO 常改动参数
parser.add_argument('--epochs',default=100,type=int)
parser.add_argument('--file_path',default='data/000001SH_index.csv',type=str)
parser.add_argument('--gpu', default=0, type=int) # gpu 卡号
parser.add_argument('--useGPU', default=False, type=bool) #是否使用GPU
parser.add_argument('---sequence_length',default=5,type = int)
parser.add_argument('--batch_first', default=True, type=bool) #是否将batch_size放在第一维
parser.add_argument('--dropout', default=0.1, type=float)
parser.add_argument('--save_file', default='model/stock.pkl') # 模型保存位置
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--layers', default=2, type=int) # LSTM层数
parser.add_argument('--input_size', default=7, type=int) #输入特征的维度
parser.add_argument('--hidden_size', default=32, type=int) #隐藏层的维度
parser.add_argument('--lr', default=0.0001, type=float) #learning rate 学习率
args = parser.parse_args()
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.useGPU else "cpu")
args.device = device
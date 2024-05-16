# @Time : 2024-05-16 15:22
# @Author : wyj
# @File : predict.py
# @Describe :
import numpy as np

from model.LSTM import mylstm
from parser_my import args
import torch
from dataset import getData
def predict():
    model = mylstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers, output_size=1)
    model.to(args.device)
    checkpoint = torch.load(args.save_file)
    model.load_state_dict(checkpoint['state_dict'])
    labels = []
    preds = []
    close_max, close_min, train_loader, test_loader = getData(args.batch_size,args.file_path)
    for _,(data,label) in enumerate(test_loader):
        pred = model(data.squeeze(1))
        list = pred.squeeze(1).tolist()
        preds.extend(list[-1])
        labels.extend(label.tolist())
    for i in range(len(preds)):
        print('预测值是%.2f,真实值是%.2f' %
              (preds[i][0] * (close_max - close_min) + close_min, labels[i][0] * (close_max - close_min) + close_min))
predict()
# @Time : 2024-05-16 12:30
# @Author : wyj
# @File : LSTM.py
# @Describe :
import torch.nn as nn
"""
batch_first为True:
    输入:shape=(batch_size,time_steps,embedding)
    输出:shape=(batch_size,time_steps,hidden)
batch_first为False:
    输入:shape=(time_steps,batch_size,embedding)
    输出:shape=(time_steps,batch_size,hidden)

    在本例中embebbing可以理解为样本的特征数
"""

class mylstm(nn.Module):
    def __init__(self,input_size=7, hidden_size=32, num_layers=1 , output_size=1 , dropout=0, batch_first=True):
        super(mylstm, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, dropout=self.dropout)
        #接一个全连接层输出最终结果
        self.fc = nn.Linear(self.hidden_size,self.output_size)
    def forward(self,x):
        out, (hidden, cell) = self.lstm(x)
        return self.fc(hidden)
model = mylstm()
# @Time : 2024-05-16 12:31
# @Author : wyj
# @File : dataset.py
# @Describe : 用来加载数据集
import numpy as np
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from parser_my import args
from torchvision import transforms
def getData(batchsize,file_path = args.file_path):
    data = pd.read_csv(file_path)
    data.drop(['ts_code','id','pre_close','trade_date'],axis = 1,inplace=True)
    #收盘价的最大和最小值
    max_data = data['close'].max(axis = 0)
    min_data = data['close'].min(axis = 0)
    #最大最小化处理
    data = data.apply(lambda x:(x - x.min())/(x.max() - x.min()))
    # 构造X和Y
    # 根据前n天的数据，预测未来一天的收盘价(close)， 例如：根据1月1日、1月2日、1月3日、1月4日、1月5日的数据（每一天的数据包含8个特征），预测1月6日的收盘价。
    sequence = args.sequence_length
    X = []
    Y = []
    for i in range(data.shape[0] - sequence):
        X.append(np.array(data.iloc[i:i+sequence,1:].values,dtype=np.float32))
        Y.append(np.array(data.iloc[[i+sequence],0].values,dtype=np.float32))
    X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size = 0.3, random_state = 42)
    train_loader = DataLoader(dataset = MyDataset(X_train,y_train,transform=transforms.ToTensor()),batch_size=batchsize,shuffle=True)
    test_loader = DataLoader(dataset=MyDataset(X_test, y_test,transform=transforms.ToTensor()), batch_size=batchsize, shuffle=True)
    return max_data, min_data, train_loader, test_loader
class MyDataset(Dataset):
    def __init__(self,x,y,transform=None):
        self.x = x
        self.y = y
        self.transform = transform
    def __getitem__(self,index):
        x1 = self.x[index]
        y1 = self.y[index]
        if self.transform:
            return self.transform(x1),y1
        return x1,y1
    def __len__(self):
        return len(self.x)


getData(16,"data/000001SH_index.csv")

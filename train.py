# @Time : 2024-05-16 13:59
# @Author : wyj
# @File : train.py
# @Describe :
import matplotlib
matplotlib.use('TkAgg')
import torch.optim
from dataset import getData
from parser_my import args
from model.LSTM import mylstm
from torch import nn
import matplotlib.pyplot as plt
def train():
    model = mylstm(input_size=args.input_size, hidden_size=args.hidden_size, num_layers=args.layers , output_size=1, dropout=args.dropout, batch_first=args.batch_first )
    model.to(args.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
    close_max, close_min, train_loader, test_loader = getData(args.batch_size,args.file_path )
    l = []
    for i in range(100):
        total_loss = 0
        for idx,(data,label) in  enumerate(train_loader):
            if args.useGPU:
                data1 = data.squeeze(1).cuda()
                pred = model(data1.cuda())
                # print(pred.shape)
                pred = pred[1, :, :]
                # print(label.shape)
            else:
                data1 = data.squeeze(1)
                pred = model(data1)
                pred = pred[1, :, :]
            loss = criterion(pred,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        l.append(total_loss)
    torch.save({'state_dict': model.state_dict()}, args.save_file)
    x_train_loss = range(100)  # loss的数量，即x轴
    plt.figure()
    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')  # x轴标签
    plt.ylabel('loss')  # y轴标签

    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x_train_loss, l, linewidth=1, linestyle="solid", label="train loss")
    plt.legend()
    plt.title('Loss curve')
    plt.show()
    plt.savefig("loss.png")
train()
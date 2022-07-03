import torch
import time
from CNNnet import CNNnet
from torch import nn


if __name__ == '__main__':
    data = torch.rand(512, 1, 1024)
    data = data.float()
    label = torch.zeros(512)
    label = label.long()

    model = CNNnet()
    print(model)
    criterion = nn.CrossEntropyLoss()
    optmizer = torch.optim.Adam(model.parameters(), lr=0.001)

    tic = time.time()

    for epoch in range(20):
        model.train()
        output = model(data)
        print(output.shape, label.shape)
        loss = criterion(output, label)
        optmizer.zero_grad()
        loss.backward()
        optmizer.step()
        print('epoch: %d, loss: %2f' % (epoch, loss.item()))
    toc = time.time()
    shijian = toc - tic
    print('模型训练时长：%.4f' % shijian)

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

data_path=r'C:\Users\BCK10G_B\Desktop\PYTHON\hour.csv'
rides = pd.read_csv(data_path)
rides.head()
#訓練模型
#取出最後一列,觀察前50條
counts = rides['cnt'][:50]
#x變量
x = torch.tensor(np.arange(len(counts), dtype = float) / len(counts), requires_grad = True)
#y變量
y = torch.tensor(np.array(counts, dtype = float), requires_grad = True)

sz=10
#初始化weights,bias
weights = torch.randn((1, sz), dtype = torch.double, requires_grad = True) 
biases = torch.randn(sz, dtype = torch.double, requires_grad = True) 
weights2 = torch.randn((sz, 1), dtype = torch.double, requires_grad = True) 

learning_rate = 0.001 
losses = []
#轉換維度
x = x.view(50, -1)
#轉換維度
y = y.view(50, -1)

for i in range(100000):
    #從輸入曾到隱含層
    hidden = x * weights + biases
    #激活函數應用
    hidden = torch.sigmoid(hidden)
    #最終預測值
    predictions = hidden.mm(weights2)# + biases2.expand_as(y)
    #計算mse
    loss = torch.mean((predictions - y) ** 2) 
    losses.append(loss.data.numpy())
    
    if i % 10000 == 0:
        print('loss:', loss)        
    loss.backward()
    #更新data數值
    weights.data.add_(- learning_rate * weights.grad.data)  
    biases.data.add_(- learning_rate * biases.grad.data)
    weights2.data.add_(- learning_rate * weights2.grad.data)    
    # 清空所有變量的梯度值。
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights2.grad.data.zero_()
    
# 打印誤差曲線
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

x_data = x.data.numpy() #
plt.figure(figsize = (10, 7))
xplot, = plt.plot(x_data, y.data.numpy(), 'o') 
yplot, = plt.plot(x_data, predictions.data.numpy())  
plt.xlabel('X')
plt.ylabel('Y')
plt.legend([xplot, yplot],['Data', 'Prediction'])
plt.show()

#預測模型

counts_predict = rides['cnt'][50:100] #預測剩下50個點


x = torch.tensor((np.arange(50, 100, dtype = float) / len(counts)), requires_grad = True)

y = torch.tensor(np.array(counts_predict, dtype = float), requires_grad = True)

x = x.view(50, -1)
y = y.view(50, -1)

hidden = x * weights + biases


hidden = torch.sigmoid(hidden)


predictions = hidden.mm(weights2)

loss = torch.mean((predictions - y) ** 2) 
print(loss)


x_data = x.data.numpy() 
plt.figure(figsize = (10, 7)) 
xplot, = plt.plot(x_data, y.data.numpy(), 'o') 
yplot, = plt.plot(x_data, predictions.data.numpy())  
plt.xlabel('X') 
plt.ylabel('Y') 
plt.legend([xplot, yplot],['Data', 'Prediction']) 
plt.show()
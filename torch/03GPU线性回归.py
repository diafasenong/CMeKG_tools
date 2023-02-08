import torch
from torch import nn
from torch import optim
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 0、准备数据
x = torch.rand([500,1]).to(device)
y_true = x*3 + 0.8

# 1、自定义模型
class MyLinear(nn.Module):
    
    def __init__(self) -> None:
        super(MyLinear,self).__init__()
        self.linear = nn.Linear(1,1)
        
    def forward(self,x):
        out = self.linear(x)
        return out

# 2、实例化
model = MyLinear().to(device)
optimizer = optim.Adam(model.parameters(),lr=1e-3)
lossfn = nn.MSELoss()

print('*'*10)
print(1e-3)

# 3、训练
for i in range(10000):
    # 计算预测值
    y_predict = model(x)
    # 计算损失
    loss = lossfn(y_predict,y_true)
    # 梯度置零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 更新参数 w b
    optimizer.step()
    if i%100 == 0:
        params = list(model.parameters())
        txt = ''
        for it in params:
            txt += ' '+str(it.item())
        print(str(loss.item())+txt)
        
# 4、评估模型
model.eval()
predict = model(x)
predict = predict.data.numpy()
plt.scatter(x.data.numpy(),y_true.data.numpy(),c='r')
plt.plot(x.data.numpy(),predict)
plt.show()
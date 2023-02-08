import torch
import matplotlib.pyplot as plt

# 1、准备数据
x = torch.rand([500,1])
y_true = x*3 + 0.8

# 学习率
lr = 0.01

# 2、通过模型
w = torch.rand([1,1],requires_grad=True)
b = torch.rand([1,1],requires_grad=True)




# 4、通过循环，反向传播，更新参数
for i in range(3000):
    # 计算y_predict
    y_predict = torch.matmul(x,w)+b
    # 3、计算loss
    loss = (y_predict-y_true).pow(2).mean()
    if w.grad is not None:
        w.grad.data.zero_()
    if b.grad is not None:
        b.grad.data.zero_()
    # 反向传播
    loss.backward()
    # 更新参数
    w.data = w.data - lr*w.grad
    b.data = b.data - lr*b.grad
    print('w,b,loss:',w.item(),b.item(),loss.item())
    
# 绘制图形
plt.figure(figsize=(20,8))
plt.scatter(x.numpy().reshape(-1),y_true.numpy().reshape(-1))
y_predict = torch.matmul(x,w)+b
plt.plot(x.numpy().reshape(-1),y_predict.detach().numpy().reshape(-1),c="r")
plt.show()
    
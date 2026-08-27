import torch
import numpy as np

# 生成一个二维张量，形状是2*3，内容在0-1 之间随机，保留四位小数
u = torch.rand(2, 3)
print(u)

# 生成一个正态分布的随机初始化矩阵，其每个值都在0附近，按正态分布的可能性进行生成
n = torch.randn(3, 3)
print(n)

# 生成一个一维张量，其中有五个值，数值在0-9的整数之间随机
i = torch.randint(0, 10, (5, ))
print(i)

a = torch.arange(0, 10, 0.5)
print(a)

# 0 - 1 之间等间隔生成五个数
l = torch.linspace(0, 1, 5)
print(l)

# 从numpy中拷贝,注意此处是零拷贝，于numpy共享内存，一者改变，两者都该改变
arr = np.array([1., 2., 3.])
t = torch.from_numpy(arr)
print(t)
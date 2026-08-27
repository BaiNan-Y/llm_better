import torch

# 创建一个全0张量
z = torch.zeros(2, 3)
print(z)

# 创建一个全1张量
o = torch.ones(2, 3)
print(o)

# 创建一个元素都一样的张量
f = torch.full((2, 3), 4)
print(f)

# 创建一个 x 维的单位张量
I = torch.eye(3)
print(I)

# 创建一个空矩阵，内容是内存中的随机遗留
e = torch.empty(2, 3)
print(e)


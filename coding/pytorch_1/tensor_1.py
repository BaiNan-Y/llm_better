import torch

ids = [101, 102, 103]
t = torch.tensor(ids)
print(t)
print(t.dtype)

t2 = torch.tensor([1, 2.5])
print(t2.dtype)

t3 = torch.tensor([1, 2], dtype=torch.float)
print(t3.dtype)
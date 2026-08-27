## Pytorch基础

以下环境建议在本机进行测试，避免算力服务器的费用开销

### Tensor操作

- 创建

torch.tensor 是所有 pytorch 后续内容的基础，为了方便后续操作 pytorch 提供的所有数据操作及后续模型逻辑之类的内容，都要基于数据类型 torch.xxx（ep: torch.int64），其中，torch 在其中进行了功能扩充，并且将传统列表转化为了类似于C++ 的高效紧凑数组

三个优点：

1. 减少内存跳转：  底层会被转换为类似于数组，连续存储在一片内存单元中，这样，在cpu进行装载时，会一次性进行装载（在 python 列表中，数组中的每一个元素都会单独占用一片空间）

2. 大幅减少内存占用：   例如一个 5个int的Tensor  5*4 字节连续存储 = 20 个字节   而一个 同样的python列表需要：5个对象 5 * 100字节 = 500 字节

3. 支持SIMD运算： 一个连续紧凑的数组支持进行SIMD运算，这种运算能够支持在矩阵运算的场景下大幅度提升运算效率，其核心原理是  一条指令处理多条数据

   普通指令（SCALAR，标量）： │ 取a[0]×b[0] │ 取a[1]×b[1] │ 取a[2]×b[2] │ 取a[3]×b[3] │    SIMD 指令（VECTOR，向量）： │  取a[0..3]×b[0..3]  │     指令1（一次搞定4对））

   但其要求必须在连续的存储空间上才能进行这种操作：（简单的深入了解：SSE 指令集：128 位 = 一次处理 **4 个** float（32位×4）、AVX2 指令集：256 位 = 一次处理 **8 个** float、AVX-512 指令集：512 位 = 一次处理 **16 个** float），从这里看，以SSE指令集为例，速度会快4倍左右，但实际上，python 还要对指令和对象存储位置进行进一步处理，实际速度差距可能在100-1000倍左右

```python
import torch

ids = [101, 102, 103]
t = torch.tensor(ids)
print(t)
print(t.dtype)

t2 = torch.tensor([1, 2.5])
print(t2.dtype)

t3 = torch.tensor([1, 2], dtype=torch.float)
print(t3.dtype)
```

故而这个torch的底层还是做了很多事的，但在实际生产中，我们会更多的使用 torch.from_numpy，其原理是一样的。

- 六种常规创建方式

```python
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

```

- 其他创建方式

随机

```
import torch

# 生成一个二维张量，形状是2*3，内容在0-1 之间随机，保留四位小数
u = torch.rand(2, 3)
print(u)

# 生成一个正态分布的随机初始化矩阵，其每个值都在0附近，按正态分布的可能性进行生成
n = torch.randn(3, 3)
print(n)

# 生成一个一维张量，其中有五个值，数值在0-9的整数之间随机
i = torch.randint(0, 10, (5, ))
print(i)
```

等差

```
# 起始数、结尾数、步长
a = torch.arange(0, 10, 0.5)
print(a)
```

等间隔

```
# 0 - 1 之间等间隔生成五个数
l = torch.linspace(0, 1, 5)
print(l)
```

from_numpy

```
# 从numpy中拷贝,注意此处是零拷贝，于numpy共享内存，一者改变，两者都该改变
arr = np.array([1., 2., 3.])
t = torch.from_numpy(arr)
print(t)
```

### 运算

- 四则运算

```
# 加
print(a + b)                    # tensor([11., 22., 33.])
print(torch.add(a, b))          # tensor([11., 22., 33.])

# 减
print(a - b)                    # tensor([ -9., -18., -27.])
print(torch.sub(a, b))          # tensor([ -9., -18., -27.])

# 乘（逐元素：a[0]*b[0]，a[1]*b[1]...）
print(a * b)                    # tensor([10., 40., 90.])
print(torch.mul(a, b))          # tensor([10., 40., 90.])

# 除
print(a / b)                    # tensor([0.1, 0.1, 0.1])
print(torch.div(a, b))          # tensor([0.1, 0.1, 0.1])
```

- 数学函数

```
print(torch.pow(a, 2))     # 每个元素平方 → tensor([1., 4., 9.])
print(a ** 2)              # 等价写法 → tensor([1., 4., 9.])

x = torch.tensor([0., 1.])
print(torch.exp(x))        # e的x次方 → tensor([1.0000, 2.7183])
print(torch.log(x))        # ln(x)    → tensor([-inf, 0.0000])

print(torch.sqrt(a))       # 开根号 → tensor([1., 1.4142, 1.7321])
print(torch.abs(torch.tensor([-1., 2., -3.])))  # 绝对值 → tensor([1., 2., 3.])
```

- 激活函数

```
# ReLU：负数变成0，正数不变
print(torch.relu(torch.tensor([-2., 0., 3.])))   # tensor([0., 0., 3.])

# Sigmoid：把任意数压到0~1之间
print(torch.sigmoid(torch.tensor([0., 2.])))     # tensor([0.5, 0.8808])

# tanh：把任意数压到-1~1之间
print(torch.tanh(torch.tensor([0., 1.])))        # tensor([0., 0.7616])
```

- softmax

```
x = torch.tensor([2.0, 1.0, 0.1])
p = torch.softmax(x, dim=-1)
print(p)                     # tensor([0.6590, 0.2424, 0.0986])
print(p.sum())               # tensor(1.0000)  ← 加起来等于1，才是概率
```

- clamp

```1
x = torch.tensor([-5., 0.5, 10.])
print(torch.clamp(x, min=0, max=1))   # tensor([0., 0.5, 1.])
# 小于0的变0，大于1的变1，中间的保持不变
```

### 矩阵运算



 


















































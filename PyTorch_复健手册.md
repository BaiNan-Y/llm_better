# PyTorch 复健手册：Tensor 操作 · 形状变换 · 自动求导

> 面向有基础、需要系统复习的开发者。三大知识点各自按「**内容 → 注意重点 → 应用方向及举例 → 实操用例（贴近生产）**」展开。
> 代码均为可运行片段（需 `pip install torch`），生产用例可直接复制改造。

## 0. 环境与通用约定

```python
import torch
print(torch.__version__)            # 2.x
print(torch.cuda.is_available())    # 生产环境一般 True

# 统一设备 + 随机种子（生产必做，保证可复现）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# 约定：模型参数 float32；分类标签 int64；输入统一 .to(device)
```

---

## 一、Tensor 操作

### 1.1 内容：核心 API 一览

**创建 Tensor**

```python
torch.tensor([1, 2, 3])          # 从 Python 数据创建（会拷贝）
torch.zeros(2, 3)                # 全 0；同类还有 ones / full / eye / empty
torch.rand(2, 3)                 # 均匀分布 U[0,1)；randn 标准正态；randint 整数
torch.arange(0, 10, 2)           # [0, 2, 4, 6, 8]
torch.linspace(0, 1, 5)          # [0, 0.25, 0.5, 0.75, 1]
torch.from_numpy(np_arr)         # 与 numpy 共享内存（注意！改 numpy 会同步改 tensor）
torch.tensor(5.0)                # 0 维标量 tensor，形状 torch.Size([])
```

**关键属性**

```python
t.dtype    # torch.float32 / torch.int64 / torch.bool ...
t.shape    # torch.Size([2, 3])
t.device   # cpu / cuda:0
t.requires_grad  # 是否纳入计算图（详见第三部分）
```

**逐元素运算**

```python
a + b / torch.add(a, b)          # 加
a - b / torch.sub(a, b)          # 减
a * b / torch.mul(a, b)          # 逐元素乘（不是矩阵乘！）
a / b / torch.div(a, b)          # 除
torch.pow(a, 2)                  # 幂
torch.exp(x); torch.log(x); torch.sqrt(x); torch.abs(x)
torch.relu(x); torch.sigmoid(x); torch.tanh(x)
torch.softmax(x, dim=-1)         # 概率归一
torch.clamp(x, min=0, max=1)     # 裁剪，数值稳定性的常用手段
```

**矩阵运算**

```python
A @ B            # matmul，支持广播；2D 即矩阵乘
torch.mm(A, B)       # 仅限 2D
torch.bmm(A, B)      # batch 矩阵乘：A(B,n,k) @ B(B,k,m) -> (B,n,m)
torch.einsum("bij,bjk->bik", A, B)   # 爱因斯坦求和，通用写法

import torch

torch.manual_seed(0)

# ========== 同一份数据，四种写法，输出完全一样 ==========
A = torch.randn(2, 3, 4)   # 3D
B = torch.randn(2, 4, 5)

print((A @ B).shape)                         # [2,3,5]
print(torch.matmul(A, B).shape)              # [2,3,5]  等价于 @
print(torch.bmm(A, B).shape)                 # [2,3,5]  batch 相等时也等价

# einsum: "bij,bjk->bik" = 批量矩阵乘
print(torch.einsum("bij,bjk->bik", A, B).shape)  # [2,3,5]

# ========== 体会限制：取消注释，亲眼看报错 ==========
x = torch.randn(3, 4)
y = torch.randn(4, 5)
print(torch.mm(x, y).shape)                  # [3,5]  2D 下 mm 好用

# torch.mm(A, B)          # 报错：mm 只认 2D
# torch.bmm(torch.randn(1,3,4), B)  # 报错：bmm 不支持广播（batch=1 也不行）

# ========== 体会 einsum 比 @ 更通用 ==========
M = torch.randn(3, 4)
print(torch.einsum("ij->ji", M).shape)       # [4,3]  转置（@ 做不到）
v = torch.randn(4)
print(torch.einsum("i,i->", v, v).shape)     # []     内积（@ 做不到）

# ========== 用 arange 手动核对数值（可选） ==========
A2 = torch.arange(6.).reshape(2, 3)
B2 = torch.arange(6.).reshape(3, 2)
print(A2 @ B2)          # 自己手算一遍，体会 C[i,j]=Σ A[i,k]*B[k,j]
```

**归约 / 统计**

```python
x.sum() / x.sum(dim=1, keepdim=True)   # 注意 dim 与 keepdim 的配合
x.mean() / x.var(unbiased=False) / x.std()
x.max(dim=1)        # 返回 (values, indices) 元组
x.argmax(dim=1)     # 分类任务"预测"这一步就是用这里
torch.norm(x, p=2)  # L2 范数（梯度裁剪内部也是它）
```

**索引与切片（生产中最高频的坑之一）**

```python
x[0]; x[:, 1]; x[..., -1]          # 切片 -> 返回视图，与原 tensor 共享内存
x[torch.tensor([0, 2])]            # 高级索引 -> 返回拷贝
mask = x > 0; x[mask]              # 布尔掩码索引
torch.nonzero(mask)                # 满足条件的坐标
torch.where(cond, a, b)            # 条件选择，比循环快得多
```

**拼接 / 拆分 / 堆叠**

```python
torch.cat([a, b], dim=0)     # 沿已有维度拼接
torch.stack([a, b], dim=1)   # 新增一个维度堆叠
torch.split(x, 2, dim=0)     # 按固定长度切分
torch.chunk(x, 4, dim=0)     # 按份数均分
torch.unbind(x, dim=0)       # 拆成元组（逐个时间步处理常用）
```

**就地操作（in-place）**

```python
x.add_(1); x.mul_(2); x.clamp_(min=0); x.zero_()
# 生产默认不用 in-place，除非有明确的省内存理由
```

**随机采样**

```python
torch.manual_seed(42)                  # 采样前固定种子保证复现
torch.randn(batch, dim)                # 高斯采样（初始化 / 重参数化采样）
torch.bernoulli(probs)                 # 伯努利采样（Dropout 内部就是它）
torch.multinomial(weights, k)          # 按权重采样（策略梯度、负采样）
torch.normal(mean, std)
```

### 1.1b 逐 API 运行示例

> 每个 API 都给一个最小可运行例子，并标注实际输出。

```python
import torch
import numpy as np
torch.manual_seed(0)

# ---------- 创建 Tensor ----------
print("tensor:", torch.tensor([1, 2, 3]))                       # tensor([1, 2, 3])
print("zeros:", torch.zeros(2, 3))                              # 全 0
print("ones:", torch.ones(2))                                   # tensor([1., 1.])
print("full:", torch.full((2, 2), 7))                           # 全 7
print("eye:", torch.eye(3))                                     # 3x3 单位阵
print("empty:", torch.empty(2, 2))                              # 未初始化（垃圾值，别读！）
print("rand:", torch.rand(2, 2))                                # U[0,1)
print("randn:", torch.randn(2, 2))                              # N(0,1)
print("randint:", torch.randint(0, 10, (3,)))                   # [0,10) 整数
print("arange:", torch.arange(0, 10, 2))                        # [0, 2, 4, 6, 8]
print("linspace:", torch.linspace(0, 1, 5))                     # 5 等分
print("from_numpy:", torch.from_numpy(np.array([1., 2.])))      # 共享内存
print("0维标量:", torch.tensor(5.0), torch.tensor(5.0).shape)   # torch.Size([])

# ---------- 关键属性 ----------
t = torch.randn(2, 3)
print("dtype:", t.dtype, "| shape:", t.shape, "| device:", t.device,
      "| requires_grad:", t.requires_grad)

# ---------- 逐元素运算 ----------
a, b = torch.tensor([1., 2.]), torch.tensor([3., 4.])
print("add:", a + b, torch.add(a, b))
print("sub:", a - b, torch.sub(a, b))
print("mul(逐元素):", a * b, torch.mul(a, b))
print("div:", a / b, torch.div(a, b))
print("pow:", torch.pow(a, 2))
x = torch.tensor([-1., 0., 2.])
print("exp:", torch.exp(torch.tensor([0., 1.])))
print("log:", torch.log(torch.tensor([1., 10.])))
print("sqrt:", torch.sqrt(torch.tensor([4., 9.])))
print("abs:", torch.abs(x))
print("relu:", torch.relu(x))
print("sigmoid:", torch.sigmoid(torch.tensor([0., 2.])))
print("tanh:", torch.tanh(torch.tensor([0., 2.])))
print("softmax:", torch.softmax(torch.tensor([2., 1., 0.1]), dim=-1))
print("clamp:", torch.clamp(x, min=-0.5, max=1.0))

# ---------- 矩阵运算 ----------
A, B = torch.randn(2, 3), torch.randn(3, 2)
print("A@B:", (A @ B).shape)                                    # (2,2)
print("mm:", torch.mm(A, B).shape)
Ab, Bb = torch.randn(4, 2, 3), torch.randn(4, 3, 5)
print("bmm:", torch.bmm(Ab, Bb).shape)                          # (4,2,5)
print("einsum:", torch.einsum("bij,bjk->bik", Ab, Bb).shape)    # 同 bmm

# ---------- 归约 / 统计 ----------
m = torch.randn(2, 3)
print("sum(全部):", m.sum(), "| sum(dim=1):", m.sum(dim=1), "| keepdim:",
      m.sum(dim=1, keepdim=True).shape)                          # (2,1)
print("mean:", m.mean(), "| var:", m.var(unbiased=False), "| std:", m.std())
vals, idx = m.max(dim=1)
print("max:", vals, idx, "| argmax:", m.argmax(dim=1))
print("norm(L2):", torch.norm(m, p=2))

# ---------- 索引与切片 ----------
m = torch.arange(9).reshape(3, 3)
print("x[0]:", m[0], "| x[:,1]:", m[:, 1], "| x[...,-1]:", m[..., -1])
print("高级索引:", m[torch.tensor([0, 2])])                     # 返回拷贝
mask = m > 4
print("布尔掩码:", m[mask])
print("nonzero:", torch.nonzero(mask))
print("where:", torch.where(m > 4, torch.ones_like(m), torch.zeros_like(m)))

# ---------- 拼接 / 拆分 / 堆叠 ----------
a, b = torch.ones(2, 3), torch.zeros(2, 3)
print("cat(dim=0):", torch.cat([a, b], dim=0).shape)            # (4,3)
print("cat(dim=1):", torch.cat([a, b], dim=1).shape)            # (2,6)
print("stack(dim=1):", torch.stack([a, b], dim=1).shape)        # (2,2,3) 新增维
v = torch.arange(10)
print("split:", [s.tolist() for s in torch.split(v, 3)])        # 按3切，末尾不足
print("chunk:", [s.tolist() for s in torch.chunk(v, 4)])        # 均分4份
print("unbind:", [t.tolist() for t in torch.unbind(torch.arange(4))])

# ---------- 就地操作（in-place） ----------
t = torch.tensor([1., 2.])
t.add_(1); print("add_:", t)                                    # [2,3]
t.mul_(2);  print("mul_:", t)                                   # [4,6]
t.clamp_(min=5); print("clamp_:", t)                            # [5,6]
t.zero_();  print("zero_:", t)                                  # [0,0]

# ---------- 随机采样 ----------
torch.manual_seed(1)
print("bernoulli:", torch.bernoulli(torch.tensor([0.2, 0.8, 0.5])))
print("multinomial:", torch.multinomial(torch.tensor([1., 10., 1.]), 5, replacement=True))
print("normal:", torch.normal(mean=0.0, std=torch.ones(3)))
print("randn采样:", torch.randn(2, 3).shape)
```

### 1.1c 逐行详解：Tensor 的 7 种创建方式

> 这一节把上面 "创建 Tensor" 部分的每一行单独拆开，说明它的**含义**，并给出**一个贴合生产的使用举例**。

#### ① `torch.tensor([1, 2, 3])` —— 从 Python 数据创建（会拷贝）

- **含义**：把 Python 的列表 / 元组 / 嵌套列表转成 tensor。这是把普通数据"搬进" PyTorch 世界最直接的方式。
- **注意**：它**会拷贝**一份数据，与原 Python 对象不再共享内存；默认 dtype 由内容推断（整数 → `int64`，浮点 → `float32`）。
- **实际举例**：

```python
import torch

# 一个 Python 列表
ids = [101, 202, 303]
t = torch.tensor(ids)
print(t)               # tensor([101, 202, 303])
print(t.dtype)         # torch.int64（因为输入是整数）

# 浮点列表默认变 float32
t2 = torch.tensor([1.5, 2.5])
print(t2.dtype)        # torch.float32

# 指定 dtype / 显式指定在哪个设备上创建
t3 = torch.tensor([1, 2], dtype=torch.float32)
print(t3, t3.dtype)    # tensor([1., 2.]) torch.float32
```

> 生产上：数据来自 numpy 数组时优先用 `torch.from_numpy`（零拷贝）；数据是 Python 容器时才用 `torch.tensor`。

#### ② `torch.zeros(2, 3)` —— 全 0 张量（及 ones / full / eye / empty 同族）

- **含义**：创建一个形状为 `(2, 3)`、所有元素都是 0 的张量。同族有：
  - `torch.ones(shape)` 全 1；
  - `torch.full(shape, value)` 全填某个值；
  - `torch.eye(n)` 单位矩阵（对角线 1）；
  - `torch.empty(shape)` 只分配内存不初始化（内容是不可预测的垃圾值）。
- **实际举例**：

```python
z = torch.zeros(2, 3)
print(z)               # tensor([[0., 0., 0.], [0., 0., 0.]])

o = torch.ones(4)
print(o)               # tensor([1., 1., 1., 1.])

f = torch.full((2, 2), 7)
print(f)               # tensor([[7., 7.], [7., 7.]])

I = torch.eye(3)
print(I)               # 3x3 单位阵，对角线为 1

e = torch.empty(2, 2)
print(e)               # 内容随机（是残留内存，别拿去计算）
```

> 生产上：初始化**累加器 / 掩码 / bias 常量**常用 `zeros`；注意力 mask 填 `-inf` 用 `full(-inf)`；`eye` 用来生成 one-hot 或构造单位权重。

#### ③ `torch.rand(2, 3)` —— 均匀分布随机（及 randn / randint 同族）

- **含义**：按**均匀分布 U[0,1)** 生成随机数，形状 `(2,3)`。同族：
  - `torch.randn(shape)` 标准**正态分布** N(0,1)（神经网络权重初始化最常用）；
  - `torch.randint(lo, hi, shape)` 整数范围内的**均匀随机整数**。
- **实际举例**：

```python
u = torch.rand(2, 3)      # 值都在 [0,1) 内
print(u)

n = torch.randn(3, 3)     # 值集中在 0 附近，正负都有（可负值）
print(n)

i = torch.randint(0, 10, (5,))   # 5 个 0~9 的整数
print(i)
```

> 生产上：`randn` 是初始化权重、做重参数化采样（VAE 采样噪声）的标配；`randint` 常用于构造测试数据 / 随机丢弃样本。

#### ④ `torch.arange(0, 10, 2)` —— 等差数列

- **含义**：`arange(start, end, step)`，生成 `[start, end)` 之间、步长为 `step` 的数列（**不含 end**）。类似 Python 的 `range`，但结果是 tensor。
- **实际举例**：

```python
a = torch.arange(0, 10, 2)
print(a)               # tensor([0, 2, 4, 6, 8])

# 默认 step=1，省略 start 默认从 0 开始
print(torch.arange(5))          # tensor([0, 1, 2, 3, 4])
print(torch.arange(1, 4))       # tensor([1, 2, 3])

# 浮点步长也支持
print(torch.arange(0, 1, 0.3))  # tensor([0.0000, 0.3000, 0.6000, 0.9000])
```

> 生产上：`arange` 常用来生成 batch 内样本的序号索引（如 `torch.arange(batch_size)`），再配合 `gather/scatter` 做样本挑选。

#### ⑤ `torch.linspace(0, 1, 5)` —— 等间隔生成 5 个数

- **含义**：`linspace(start, end, steps)`，在 `[start, end]` **区间内平均切分成 steps 个点**，**包含首尾两端**。和 `arange` 的关键区别：`arange` 指定的是"步长"，`linspace` 指定的是"个数"。
- **实际举例**：

```python
l = torch.linspace(0, 1, 5)      # 0 到 1 平均 5 个点（含 0 和 1）
print(l)                          # tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])

# 常用：生成余弦学习率衰减 / 位置编码的分母
import math
freq = torch.linspace(0, 0.5, 4)   # 4 个频率
print(freq)
```

> 生产上：**Transformer 位置编码**里 `torch.linspace(0, log(max_len/2), d_model//2)` 生成频率表；画图 / 曲线采样也用 `linspace`。

#### ⑥ `torch.from_numpy(np_arr)` —— 与 numpy 共享内存（零拷贝）

- **含义**：把 numpy 数组**包装**成 tensor。关键点：**不拷贝，二者共享同一块内存**——改其中一个，另一个也会变。想隔离就用 `.clone()`。
- **实际举例**：

```python
import numpy as np
arr = np.array([1., 2., 3.])
t = torch.from_numpy(arr)

print(t)                 # tensor([1., 2., 3.], dtype=torch.float64)

# 改 numpy -> tensor 跟着变（共享内存）
arr[0] = 100
print(t)                 # tensor([100., 2., 3.])，变了！

# 反之亦然
t[1] = 999
print(arr)               # [100. 999. 3.]，也变了！
```

> 生产上：数据预处理经常用 numpy 完成（读文件、pandas 清洗），随后 `torch.from_numpy` 零拷贝接入模型——注意内存共享，需要独立副本时 `.clone()`。**另外**：numpy 默认 `float64`，转成 tensor 后如需 float32 要 `.float()`。

#### ⑦ `torch.tensor(5.0)` —— 0 维标量 tensor

- **含义**：用**单个数值**创建的 0 维 tensor（形状是 `torch.Size([])`），表示一个标量。它仍是一个 tensor，但没有任何维度。
- **实际举例**：

```python
s = torch.tensor(5.0)
print(s)               # tensor(5.)
print(s.shape)         # torch.Size([])  <- 0 维
print(s.dim())         # 0

# 取出 Python 数值：.item()
print(s.item())        # 5.0，纯 Python float

# 标量 tensor 可以做正常的算术
print(s + torch.tensor(1.0))   # tensor(6.)
```

> 生产上：`loss.item()`、`metric.item()` 就是把标量 tensor 拿出来记日志 / 画曲线的标准姿势；一个函数只返回"一个数字"时也用 0 维 tensor 承载。

### 1.2 注意重点

1. **`*` 是逐元素乘，矩阵乘必须用 `@` / `matmul` / `mm` / `bmm`。** 新手最易错，没有之一。
2. **广播（broadcasting）从右往左对齐**，维度不匹配时必须是 1 或缺失。
   - `(3,1) + (2,)` 报错（1 vs 2 不兼容）；`(3,1) + (1,2)` -> `(3,2)` 合法。
3. **切片返回视图（共享内存），高级索引 / 布尔索引返回拷贝。** 对视图 in-place 修改会污染原 tensor，这是"数据悄悄变掉"类 bug 的常见来源。
4. **默认 dtype 是 float32**；`torch.tensor` 从 Python 整数创建默认 int64；神经网络中 float16 只在混合精度时出现。
5. **0 维标量 tensor**：取 Python 数值用 `.item()`（打日志 / 写 metric）；想保持 tensor 形状用 `squeeze()` / `reshape(())`。
6. **CPU/GPU 混用直接报错**：`Expected all tensors to be on the same device`。习惯在入口统一 `.to(device)`。
7. **in-place 运算会破坏 autograd 计算图**（第三部分详述），且 `x += 1` 等价于 `x.add_(1)`，容易误触。
8. **数值稳定性优先**：用 `torch.log_softmax`、`F.logsigmoid`、`clamp(1e-8)` 而不是先算概率再取 log（会下溢成 0/log(0)）。
9. **`from_numpy` 共享内存**：与 numpy 数组同源，需要隔离时 `.clone()`。
10. **复现三件套**：`torch.manual_seed` + `numpy.random.seed` + `random.seed` 一起固定，GPU 还要 `torch.cuda.manual_seed_all`。

### 1.3 应用方向及举例

| 应用方向 | 典型场景 | 涉及 API |
|---|---|---|
| 数据预处理 | 标准化、归一化、缺失值填充 | `mean/std/clamp/where` |
| 特征工程 | 数值分箱、One-Hot、交互特征 | `searchsorted/scatter_/eye` |
| 损失计算 | 交叉熵、MSE、自定义损失 | `gather/log/softmax` |
| 推荐 / 检索 | 向量内积、余弦相似度、top-k | `matmul/norm/topk` |
| 采样 | 负采样、策略梯度采样 | `multinomial/bernoulli` |

### 1.4 实操用例（贴近生产）

**用例 1.4.1 数据管道：标准化 + 异常值裁剪**

生产要点：mean/std 必须来自训练集统计，禁止用当前 batch 现算（否则 train/serving 不一致）。

```python
def normalize_feature(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                      lower: float = -3.0, upper: float = 3.0) -> torch.Tensor:
    """z-score 标准化，并把极端值裁剪到 ±3 个标准差。"""
    x = (x - mean) / (std + 1e-8)          # +1e-8 防除零（生产惯例）
    return torch.clamp(x, min=lower, max=upper)

# 训练集统计（生产上由数据管线预先算好并持久化）
x_train = torch.randn(10000, 8) * torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.])
mean, std = x_train.mean(0), x_train.std(0)

# 线上一个 batch，含离群点 100
x_live = torch.tensor([[100., -0.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
print(normalize_feature(x_live, mean, std))  # 100 被压到 3.0
```

**用例 1.4.2 推荐召回：批量余弦相似度 + top-k**

生产要点：一次矩阵乘算完所有 query 与全部 item 的相似度，避免 Python 循环。

```python
def batch_topk(query: torch.Tensor, item_emb: torch.Tensor, k: int = 10):
    """query: (B, D)，item_emb: (N, D)。返回每个 query 的 top-k 索引与分数。"""
    q = torch.nn.functional.normalize(query, dim=-1)    # (B, D)
    m = torch.nn.functional.normalize(item_emb, dim=-1) # (N, D)
    scores = q @ m.T                                     # (B, N)
    return torch.topk(scores, k=k, dim=-1)

q = torch.randn(32, 128); items = torch.randn(10_000, 128)
idx, score = batch_topk(q, items)
print(idx.shape, score.shape)   # (32, 10) (32, 10)
```

---

## 二、形状变换（Tensor Reshaping）

### 2.1 内容：核心 API 一览

```python
x.view(2, 3)              # 视图：要求内存连续；-1 自动推断
x.reshape(2, 3)           # 视图优先，不连续时自动拷贝（生产更推荐）
x.T                       # 转置（视图，内存不连续）
x.transpose(1, 2)         # 交换两个维度（视图）
x.permute(2, 0, 1)        # 任意维度重排（视图），HWC -> CHW 就用它
x.squeeze() / x.squeeze(1)    # 去掉 size=1 的维度（强烈建议指定 dim）
x.unsqueeze(0) / x.unsqueeze(-1)  # 指定位置插入 size=1 维度
x.flatten(1, -1)          # 保留前 1 维，其余展平（全连接层输入的标准姿势）
x.contiguous()            # 强制拷贝为连续内存
x.expand(4, 3)            # 广播式扩展（视图，不分配新内存！）
x.repeat(2, 1)            # 真实复制扩展（量大时小心 OOM）
torch.cat / torch.stack   # 沿已有维拼接 / 新增维堆叠（见 1.1）
torch.split / torch.chunk / torch.unbind   # 切分（见 1.1）

# 进阶：einsum 替代一连串 transpose + matmul
out = torch.einsum("bi,ij,bj->b", x, A, x)   # 每个样本的二次型 x^T A x

# 与 numpy / dtype / device 互转
x.numpy()                 # 需先 .detach().cpu()（有梯度的 tensor 直接调会报错）
torch.from_numpy(arr)
x.to(dtype=torch.float16, device="cuda")
```

### 2.1b 逐 API 运行示例

```python
import torch
torch.manual_seed(0)

x = torch.arange(6).reshape(2, 3)          # [[0,1,2],[3,4,5]]

# view：连续内存可用，-1 自动推断
print("view:", x.view(3, 2).tolist())
print("view(-1):", x.view(-1).tolist())    # 展平 [0..5]
# reshape：不连续时自动拷贝，更安全
y = x.t()                                  # transpose 后的视图（不连续）
print("reshape非连续:", y.reshape(6).tolist())
# print(y.view(6))                         # 报错！不连续不能 view

# T / transpose / permute
print("T:", x.T.tolist())                  # [[0,3],[1,4],[2,5]]
print("transpose(0,1):", x.transpose(0, 1).shape)   # (3,2)
c = torch.arange(24).reshape(2, 3, 4)
print("permute(2,0,1):", c.permute(2, 0, 1).shape) # (4,2,3)

# squeeze / unsqueeze
s = torch.zeros(1, 3, 1)
print("squeeze():", s.squeeze().shape)     # (3,) 全去
print("squeeze(0):", s.squeeze(0).shape)   # (3,1)
print("unsqueeze(0):", x.unsqueeze(0).shape)   # (1,2,3)
print("unsqueeze(-1):", x.unsqueeze(-1).shape) # (2,3,1)

# flatten
print("flatten(1,-1):", x.flatten(1, -1).shape)  # (2,3) 保留batch
print("flatten():", x.flatten().shape)           # (6,)

# contiguous
y = x.t()                                  # 不连续
print("is_contiguous:", y.is_contiguous()) # False
print("contiguous():", y.contiguous().is_contiguous())  # True

# expand / repeat（关键区别：内存）
e = torch.tensor([[1.], [2.]])             # (2,1)
print("expand:", e.expand(2, 3).tolist())  # [[1,1,1],[2,2,2]] 视图
print("repeat:", e.repeat(1, 3).tolist())  # 同上但真复制
# e.expand(-1, 3) 等价写法：-1 表示保持原大小

# cat / stack
a, b = torch.ones(2, 3), torch.zeros(2, 3)
print("cat:", torch.cat([a, b], dim=0).shape)     # (4,3)
print("stack:", torch.stack([a, b], dim=1).shape) # (2,2,3)

# split / chunk / unbind
v = torch.arange(10)
print("split:", [t.tolist() for t in torch.split(v, 3)])
print("chunk:", [t.tolist() for t in torch.chunk(v, 4)])
print("unbind:", [t.tolist() for t in torch.unbind(v)])

# einsum：二次型
x1, A1 = torch.randn(3), torch.randn(3, 3)
print("einsum 二次型:", torch.einsum("i,ij,j->", x1, A1))
print("等价:", (x1 @ A1 @ x1))

# numpy / from_numpy / to
tn = torch.arange(3)
arr = tn.numpy()                           # CPU 且无梯度才可直接 numpy
print("numpy:", arr)                       # [0 1 2]
arr[0] = 99                                # 改 numpy 会同步改 tensor（共享内存）
print("共享内存(tensor被改):", tn)          # tensor([99, 1, 2])
tn2 = torch.from_numpy(arr)                # 反向共享
print("from_numpy:", tn2)
f16 = tn2.to(dtype=torch.float16)
print("to(float16):", f16.dtype)           # torch.float16
```

### 2.2 注意重点

1. **`view` 要求内存连续**：`transpose` / `permute` 之后再用 `view` 直接报错。生产上无脑用 `reshape`，或 `view` 前先 `.contiguous()`。
2. **视图共享内存**：`view/reshape/transpose` 的结果与原 tensor 同源，in-place 修改会传染。要独立副本用 `.clone()`。
3. **`expand` 不分配内存**（stride=0 技巧），省内存但后续任何要求"真数据"的操作（如 `contiguous()`、`numpy()`）会触发真实拷贝；`repeat` 是真复制，数据量大时显存翻倍。
4. **`transpose` 只交换两个维度，`permute` 可任意排列**，二者都返回非连续视图。
5. **维度语义是生产 bug 大户**：CV 里 `(B,C,H,W)` 与 `(B,H,W,C)` 搞混；NLP/Transformer 里 `(B,S,H,D)` 拆头顺序错位。用 `einsum` 或注释固定维度名可减少此类问题。
6. **`squeeze()` 不指定 dim 会把所有 size=1 的维都去掉**，batch 恰为 1 时形状悄然改变，下游全炸；尽量 `squeeze(dim)`。
7. **`flatten(1, -1)` 保留 batch 维**是进全连接层的标准姿势；`view(-1)` 是整体展平，两者别混。
8. **`reshape` 与 numpy 的 `reshape` 语义不同**：PyTorch 的 reshape 对不连续 tensor 可能返回拷贝，但结果数据顺序始终正确——放心用。
9. **`cat` 不增维（沿已有维拼接），`stack` 增一维堆叠**：组 batch 用 `stack`，特征拼接用 `cat`。
10. **`permute` 后 `shape` 与 `stride` 要一起想**：打印 `.is_contiguous()` 确认内存布局，性能敏感路径尤其重要。

### 2.3 应用方向及举例

| 应用方向 | 场景 | 关键 API |
|---|---|---|
| 图像处理 | HWC -> CHW、归一化、数据增强 | `permute/contiguous/view` |
| 文本 / NLP | 组装 batch、mask 扩展、位置编码 | `stack/expand` |
| Transformer | 多头注意力拆头 / 合并头 | `view/transpose/reshape` |
| 全连接分类头 | 展平进入 FC | `flatten(1,-1)` |
| 序列模型 | 时间步与 batch 维度互换 | `transpose(0,1)` |
| 混合精度 / 部署 | dtype 与内存布局转换 | `.to()/contiguous` |

### 2.4 实操用例（贴近生产）

**用例 2.4.1 图像预处理管线：HWC -> CHW + 归一化**

生产要点：`permute` 得到的是非连续视图，必须 `contiguous()` 才适合进卷积；归一化均值/方差要用广播形状 `(1,3,1,1)`。

```python
def preprocess_batch(images_bhwc: torch.Tensor,
                     mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225)) -> torch.Tensor:
    """(B, H, W, 3) uint8[0,255] -> (B, 3, H, W) float32 已归一化（ImageNet 惯例）"""
    x = images_bhwc.float() / 255.0                     # 转 float 并归一化
    x = x.permute(0, 3, 1, 2).contiguous()              # (B, C, H, W)
    m = torch.tensor(mean).view(1, 3, 1, 1)             # 广播形状
    s = torch.tensor(std).view(1, 3, 1, 1)
    return (x - m) / s

imgs = torch.randint(0, 255, (4, 224, 224, 3))          # 模拟 4 张图
out = preprocess_batch(imgs)
print(out.shape, out.dtype)   # torch.Size([4, 3, 224, 224]) torch.float32
```

**用例 2.4.2 Transformer 多头注意力：拆头 -> 算注意力 -> 合并**

生产要点：`view` 拆出 head 维，`transpose` 把 head 提到第 1 维实现"批量并行"，最后 `reshape` 合并。

```python
def attention(qkv: torch.Tensor, num_heads: int):
    """qkv: (B, S, 3*H*D)。返回合并后的注意力输出 (B, S, H*D)。"""
    B, S, _ = qkv.shape
    D = (qkv.shape[-1] // 3) // num_heads
    q, k, v = torch.split(qkv, qkv.shape[-1] // 3, dim=-1)   # 各 (B, S, H*D)
    # (B, S, H, D) -> (B, H, S, D)
    q = q.view(B, S, num_heads, D).transpose(1, 2)
    k = k.view(B, S, num_heads, D).transpose(1, 2)
    v = v.view(B, S, num_heads, D).transpose(1, 2)
    scores = (q @ k.transpose(-1, -2)) / (D ** 0.5)          # (B, H, S, S)
    attn = torch.softmax(scores, dim=-1)
    out = attn @ v                                            # (B, H, S, D)
    return out.transpose(1, 2).reshape(B, S, -1)              # 合并回 (B, S, H*D)

qkv = torch.randn(2, 16, 768)   # 模拟 12 头、每头 64 维
print(attention(qkv, num_heads=12).shape)   # (2, 16, 768)
```

**用例 2.4.3 序列 batch：不定长序列 padding（NLP 生产）**

生产要点：padding 用 `full` 填充 pad 符号，配合 `lengths` 在 loss 里做 mask，别让 pad 参与计算。

```python
def collate_pad(seqs: list[torch.Tensor], pad_idx: int = 0):
    """长度不一的序列 -> (B, L_max) + 每样本真实长度。"""
    L = max(len(s) for s in seqs)
    batch = torch.full((len(seqs), L), pad_idx, dtype=torch.long)
    lengths = torch.tensor([len(s) for s in seqs])
    for i, s in enumerate(seqs):
        batch[i, :len(s)] = s
    return batch, lengths

seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
batch, lengths = collate_pad(seqs)
print(batch)     # [[1,2,3],[4,5,0],[6,0,0]]
print(lengths)   # tensor([3, 2, 1])
```

---

## 三、自动求导 autograd

### 3.1 内容：核心概念与 API

```python
x = torch.randn(3, requires_grad=True)   # 打开梯度追踪
y = (x ** 2).sum()                        # y.grad_fn = SumBackward0，动态建图
y.backward()                              # 反向传播，填充 x.grad
print(x.grad)                             # tensor([2x0, 2x1, 2x2])

# 梯度追踪的三种开关
with torch.no_grad():          # 推理 / 评估 / 特征提取：不建图、省显存
    y = model(x)
torch.inference_mode()         # no_grad 的增强版，纯推理首选（更快）
torch.enable_grad()            # 在 no_grad 块里临时恢复追踪

# 切断梯度
z = y.detach()                 # 拿到与计算图无关的视图（GAN 假样本、目标值截断）
y.detach_()                    # 就地版本

# 与 backward 相关的参数
loss.backward(retain_graph=True)   # 保留计算图，允许二次 backward（仅 debug/元学习）
loss.backward(create_graph=True)   # 建立二阶导计算图（MAML 等元学习用）

# 生产工具
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
optimizer.zero_grad(set_to_none=True)   # 标准清零写法（比 zero_ 更快更省显存）
```

**计算图与叶子节点（经典困惑点）**

```python
x = torch.randn(3, requires_grad=True)   # 叶子：由用户直接创建
y = x * 2                                 # 非叶子：中间结果
z = y.sum()
z.backward()
print(x.is_leaf, y.is_leaf)   # True False
print(x.grad)                 # tensor([2., 2., 2.])
print(y.grad)                 # None  <- 非叶子默认不保存梯度（调试可用 retain_grad()）
```

**自定义 autograd.Function（自定义算子并让梯度正确传播）**

```python
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)          # 保存反向需要的中间量
        return torch.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * (x > 0)      # ReLU 的导数是阶梯函数

y = MyOp.apply(x)
```

**其他常用能力**

```python
torch.autograd.grad(loss, params)        # 手动取梯度，不经过 .backward()（梯度惩罚用）
tensor.register_hook(...)                # 查看 / 修改某个 tensor 的梯度
model.register_full_backward_hook(...)   # 打印各层梯度范数（定位梯度消失/爆炸）
torch.autograd.set_detect_anomaly(True)  # 开发期定位 NaN 来源（很慢，生产别开）
```

### 3.1b 逐 API 运行示例

```python
import torch
torch.manual_seed(0)

# ---------- requires_grad / backward ----------
x = torch.randn(3, requires_grad=True)
y = (x ** 2).sum()
print("grad_fn:", y.grad_fn)               # SumBackward0（证明建了图）
y.backward()
print("x.grad:", x.grad)                   # tensor([2x0, 2x1, 2x2])

# ---------- no_grad / inference_mode / enable_grad ----------
x = torch.randn(1, requires_grad=True)
with torch.no_grad():
    z = x * 2
    print("no_grad 下无图:", z.requires_grad, z.grad_fn is None)  # False True
with torch.inference_mode():               # 增强版 no_grad，纯推理更快
    z2 = x * 2
    print("inference_mode 下无图:", z2.requires_grad)
with torch.enable_grad():                  # 在 no_grad 块里临时恢复
    with torch.no_grad():
        pass
    z3 = x * 2                             # 在 no_grad 外层，正常建图
    print("enable_grad 外建图:", z3.requires_grad)   # True

# ---------- detach / detach_ ----------
x = torch.randn(2, requires_grad=True)
y = (x * 2).sum()
z = y.detach()                             # 切断，与原图无关
print("detach 后 requires_grad:", z.requires_grad)   # False
z += 10                                    # 不污染计算图
print("detach 不传染:", y)                  # 仍是 (x*2).sum() 的式子

# ---------- backward 参数 ----------
x = torch.randn(3, requires_grad=True)
y = (x * 2).sum()
y.backward(retain_graph=True)              # 保留图，可再 backward
print("first:", x.grad)
x.grad.zero_()                             # 需手动清零再算
y.backward()                               # 第二次（因为上面 retain 了）
print("second:", x.grad)

x2 = torch.randn(2, requires_grad=True)
y2 = (x2 ** 2).sum()
g2 = torch.autograd.grad(y2, x2, create_graph=True)  # 二阶导
g2_flat = g2[0].sum()
g2_flat.backward()                         # 对梯度再求导 -> 二阶
print("二阶导(x2²的导数是2x，再导是2):", x2.grad)   # tensor([2., 2.])

# ---------- 梯度裁剪 / zero_grad ----------
model = torch.nn.Linear(2, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
out = model(torch.randn(2)); out.sum().backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 裁剪后
opt.step()
opt.zero_grad(set_to_none=True)            # 清空梯度
print("step 后参数已更新，梯度已清零")

# ---------- 叶子节点 / retain_grad ----------
x = torch.randn(3, requires_grad=True)     # 叶子
y = x * 2                                  # 非叶子
z = y.sum()
z.backward()
print("is_leaf:", x.is_leaf, y.is_leaf)    # True False
print("x.grad:", x.grad, "| y.grad:", y.grad)  # y.grad 为 None

w = torch.randn(3, requires_grad=True)
v = w * 2; v.retain_grad()                 # 显式保留非叶子梯度
v.sum().backward()
print("retain_grad 后 v.grad:", v.grad)    # tensor([2., 2., 2.])

# ---------- 自定义 autograd.Function ----------
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.relu(x)
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * (x > 0)

a = torch.tensor([-1., 0., 2.], requires_grad=True)
b = MyOp.apply(a).sum()
b.backward()
print("自定义 ReLU 梯度:", a.grad)          # tensor([0., 0., 1.])（x>0 才为 1）

# ---------- autograd.grad / register_hook / anomaly ----------
a = torch.randn(2, requires_grad=True)
loss = (a * a).sum()
g = torch.autograd.grad(loss, a)           # 直接取梯度，不调 backward
print("autograd.grad:", g[0])              # tensor([2a0, 2a1])

a = torch.randn(2, requires_grad=True)
hook_vals = []
a.register_hook(lambda grad: hook_vals.append(grad.clone()))
(a ** 2).sum().backward()
print("register_hook 抓到的梯度:", hook_vals)

torch.autograd.set_detect_anomaly(True)    # 开发期定位 NaN（生产勿开）
print("set_detect_anomaly 已开启")
```

### 3.2 注意重点

1. **梯度默认累加**：每次 `backward` 前必须 `zero_grad()`，否则梯度逐 batch 叠加，loss 曲线疯涨。（梯度累积技巧正是利用这一点。）
2. **只有浮点 / 复数类型能开 `requires_grad`**；整型 tensor 开启会直接报错。
3. **同一计算图重复 `backward` 会报错**："Trying to backward through the graph a second time"。正常训练不需要 `retain_graph`，出现该报错多半是代码结构问题。
4. **in-place 操作破坏计算图**：对图中 tensor 做 `add_` 可能报 `a leaf Variable that requires grad is being used in an in-place operation`，或静默算出错误梯度。
5. **非叶子节点 `.grad` 为 None**，调试需要时 `retain_grad()`。
6. **`no_grad` 是作用域开关（影响块内所有运算），`detach` 只作用于单个 tensor**。推理用 `torch.inference_mode()` 最快。
7. **验证 / 推理不包 `no_grad`**：每个 batch 都建图，显存翻倍、速度变慢，是生产事故高发点。
8. **取数标准姿势**：`.detach().cpu().numpy()`；对 requires_grad=True 的 tensor 直接 `.numpy()` 报错。
9. **梯度裁剪在 `step` 之前调用**：`clip_grad_norm_`（按范数）比 `clip_grad_value_`（按值）更常用。
10. **混合精度（AMP）顺序不能乱**：`scaler.scale(loss).backward()` -> `unscale_` -> 裁剪 -> `scaler.step()` -> `scaler.update()`，见用例 3.4.4。
11. **冻结参数**：`param.requires_grad = False` 后不建图不更新；`model.eval()` 只影响 BN/Dropout，不影响梯度——两个概念别混。
12. **`with torch.enable_grad()` 里手动对输入求梯度**：对抗攻击、可解释性（saliency map）都要对输入而非参数求梯度。

### 3.3 应用方向及举例

| 应用方向 | 场景 | 关键点 |
|---|---|---|
| 模型训练 | forward -> loss -> backward -> step | backward / zero_grad / step |
| 迁移学习 | 冻结 backbone 只训分类头 | `requires_grad=False` |
| GAN / 扩散模型 | 交替训练、假样本截断 | `detach` |
| 元学习（MAML） | 二阶梯度 | `create_graph=True` |
| 强化学习 | 策略梯度、PPO | 停止梯度 + 手动构造 loss |
| 对抗攻击 / 可解释性 | 对输入求梯度 | `enable_grad` |
| 知识蒸馏 | 教师只前向不更新 | `no_grad` |

### 3.4 实操用例（贴近生产）

**用例 3.4.1 标准训练循环（含梯度累积 + 梯度裁剪）**

生产要点：梯度累积把等效 batch 放大（显存不够时），注意 loss 要除以累积步数、且只在累积满时 step + 清零。

```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

X = torch.randn(2048, 10); y = (X[:, 0] * 2 + X[:, 1] > 0).long()
loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

ACCUM_STEPS = 2      # 梯度累积步数
MAX_NORM = 1.0       # 梯度裁剪阈值

model.train()
for epoch in range(3):
    total, n = 0.0, 0
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb) / ACCUM_STEPS
        loss.backward()                            # 梯度累加到 .grad

        if (i + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            optimizer.step()                       # 用累积后的梯度更新
            optimizer.zero_grad(set_to_none=True)

        total += loss.item() * ACCUM_STEPS
        n += xb.size(0)
    print(f"epoch {epoch}: loss={total / n:.4f}")
```

**用例 3.4.2 迁移学习：冻结特征提取器**

生产要点：`filter(lambda p: p.requires_grad, ...)` 让优化器只看到可训练参数，省显存也省时间。

```python
backbone = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
head = nn.Linear(64, 2)
model = nn.ModuleList([backbone, head])

for p in backbone.parameters():
    p.requires_grad = False                        # 冻结

opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
print("trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad))
```

**用例 3.4.3 GAN 交替训练：detach 截断梯度**

```python
z = torch.randn(64, 128)
fake = G(z)
d_loss = D(fake.detach())     # 判别器更新：梯度止于 D，不回流到 G
# ... d_loss.backward(); D_opt.step()
g_loss = D(fake)              # 生成器更新：不 detach，梯度经 D 流回 G
```

**用例 3.4.4 混合精度 + 梯度裁剪（AMP 生产模板，顺序不可乱）**

```python
scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    with torch.autocast(device_type="cuda"):
        out = model(xb)
        loss = loss_fn(out, yb)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)                     # ① 还原真实梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # ② 用真实梯度裁剪
    scaler.step(optimizer)                         # ③ scaler 内部处理缩放
    scaler.update()                                # ④ 更新缩放因子
    optimizer.zero_grad(set_to_none=True)
```

**用例 3.4.5 debug：梯度监控与 NaN 定位**

```python
torch.autograd.set_detect_anomaly(True)   # 开发期开启，报出产生 NaN 的前向位置

# 每个 step 后监控参数梯度范数（生产可挂到监控系统）
for name, p in model.named_parameters():
    if p.grad is not None:
        g = p.grad.norm().item()
        if g != g:                                   # NaN 检测（NaN 与自身不等）
            print(f"[WARN] {name} 梯度为 NaN")
```

---

## 四、综合实战：贴近生产的训练脚本骨架

把前面所有要点串起来：seed 固定、device、AMP、梯度裁剪、checkpoint、early stop、推理用 inference_mode。

```python
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x):
        return self.net(x)

model = SimpleModel().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

X = torch.randn(4096, 10); y = (X.sum(1) > 0).long()
loader = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(X, y), batch_size=256)

best_acc, patience, no_improve = 0.0, 3, 0
for epoch in range(10):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            loss = loss_fn(model(xb), yb)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()

    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    acc = correct / total
    print(f"epoch {epoch}: val_acc={acc:.4f}")

    if acc > best_acc:                 # 只在更优时存档 + 重置耐心计数
        best_acc, no_improve = acc, 0
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "acc": acc}, "best.pt")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("early stop")
            break
```

---

## 五、高频踩坑速查表

| 现象 | 原因 | 解决 |
|---|---|---|
| 训练 loss 不降 / 发散 | 忘了 `zero_grad`，梯度累加 | `opt.zero_grad(set_to_none=True)` |
| `in-place operation` 报错 | 对叶子 / 图内 tensor 做 `+=` 等 | 改 `x = x + 1` 或 `clone()` |
| `view size is not compatible` | transpose/permute 后用 view | `.contiguous()` 或改 `reshape` |
| `.numpy()` 报错 | tensor 需要梯度 / 在 GPU 上 | `.detach().cpu().numpy()` |
| 显存 OOM | 推理没包 no_grad / batch 太大 | `inference_mode` + 梯度累积 |
| 梯度或 loss 为 NaN | 除零 / 数值下溢 / 学习率过大 | 加 epsilon、`clamp`、`clip_grad_norm_` |
| 每个 epoch 结果不一致 | 未固定随机种子 | 固定 torch/numpy/random 三套种子 |
| 精度异常但代码"看起来对" | 数据 dtype 被隐式转 float32 | 检查 `to()` 与 dtype 转换位置 |
| 训练慢、显存高 | 前向没在 no_grad 外（多余建图） | 用 `inference_mode` 包推理路径 |
| 冻结后仍被更新 | 只调了 `model.eval()`（它不影响梯度） | 显式 `param.requires_grad = False` |

---

## 六、推荐练习路径

1. **Tensor 操作**：手写 z-score 标准化 -> 余弦相似度 top-k -> One-Hot 编码（`scatter_`）。
2. **形状变换**：手写 HWC->CHW 预处理 -> 手写多头注意力 reshape -> 手写 collate padding。
3. **autograd**：手写一个 2 层 MLP 的完整训练循环 -> 加梯度裁剪 -> 加 AMP -> 加 early stop。
4. **进阶挑战**：不调用 `nn`，只用 Tensor 运算实现 `Linear` + `ReLU` 的前向/反向（自证对计算图的理解）；用 `autograd.grad` 实现梯度惩罚（WGAN-GP）。


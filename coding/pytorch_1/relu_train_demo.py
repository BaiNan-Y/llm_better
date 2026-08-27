# -*- coding: utf-8 -*-
"""
【最重要的一次演示】有激活函数 vs 没有激活函数，到底差在哪？
用最最最简单的小网络，手把手看每一步怎么算。

核心问题：只有 1 个输入特征 x，目标是拟合抛物线 y = x²。
例子用不到真实网络，我们对比两个小网络：

  网络A（无激活函数）：y = w1*(w0*x + b0) + b1   （两层线性叠一起）
  网络B（有ReLU激活） ：y = w1*ReLU(w0*x + b0) + b1

分别训练它们去拟合 y = x²，看谁能成、谁不行。
"""

import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("问题背景：用一个小网络去拟合抛物线 y = x²（这是一个非线性目标）")
print("=" * 70)

# 训练数据：几个点
xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
ys = xs ** 2

print("\n训练数据 (5个点)：")
for i in range(len(xs)):
    print(f"  x={xs[i]:>4}  →  目标y={ys[i]:>4}")

# =========================================================
# 先证明一个数学事实：没有激活函数的两层线性网络 = 一层线性网络
# =========================================================
print("\n" + "=" * 70)
print("第一步：证明『没有激活函数』的网络，其实就是一条直线")
print("=" * 70)
print("""
两层线性网络的公式：
    y = w1 * (w0*x + b0) + b1
展开：
    y = w1*w0 * x + (w1*b0 + b1)
             │              │
      ┌──────┴──────┐ ┌─────┴─────┐
      │  还是斜率   │ │  还是截距  │
      └─────────────┘ └───────────┘
结论：不管 w0,b0,w1,b1 取什么值，它永远是
    y = k*x + c    （一条直线！）
永远画不出一条弯曲的抛物线。
""")

# =========================================================
# 网络A：没有激活函数 —— 手动训练试试，看它拼命也拼不直
# =========================================================
print("=" * 70)
print("网络A（没有激活函数）：开始训练，目标是拟合抛物线")
print("=" * 70)

w0, b0 = 1.0, 0.5
w1, b1 = 1.0, 0.5
lr = 0.05

def netA(x):
    return w1 * (w0 * x + b0) + b1

# 先打印训练开始时的输出
print("\n【epoch 0】初始化 w0=1.0,b0=0.5,w1=1.0,b1=0.5")
for i in range(len(xs)):
    print(f"  输入 x={xs[i]:>4}  →  网络输出 {netA(xs[i]):>6.3f}   (目标 {ys[i]:>4})")

print("\n训练 2000 轮（每轮微调 w0,b0,w1,b1）...")

for epoch in range(2000):
    # 前向：算输出
    pred = netA(xs)
    # 损失：均方误差
    loss = np.mean((pred - ys) ** 2)
    # 梯度（对每个参数求偏导，这里直接给结果）
    d_loss_d_pred = 2 * (pred - ys) / len(xs)
    d_w1 = np.sum(d_loss_d_pred * (w0 * xs + b0))
    d_b1 = np.sum(d_loss_d_pred)
    d_w0 = np.sum(d_loss_d_pred * w1 * xs)
    d_b0 = np.sum(d_loss_d_pred * w1)
    # 更新
    w0 -= lr * d_w0
    b0 -= lr * d_b0
    w1 -= lr * d_w1
    b1 -= lr * d_b1

    if epoch in [0, 1, 2, 100, 1999]:
        print(f"  epoch {epoch:<5} loss={loss:.5f}")

# 训练结束后看输出
print("\n【网络A 训练完成后】的输出：")
for i in range(len(xs)):
    print(f"  输入 x={xs[i]:>4}  →  网络输出 {netA(xs[i]):>6.3f}   (目标 {ys[i]:>4})")

# 判断它学到的是不是直线
k = w1 * w0
c = w1 * b0 + b1
print(f"\n  数学事实验证：这个网络无论怎么训练，本质都是 y = {k:.3f}x + {c:.3f}  ← 一条直线")
print("  它永远无法同时让  x=2→4 且 x=-2→4  （抛物线两端都要向上）")
print("  → 所以 loss 降不下去，这是结构性的失败，不是训练不够！")

# =========================================================
# 网络B：有 ReLU —— 关键，看它如何弯出曲线
# =========================================================
print("\n" + "=" * 70)
print("网络B（带 ReLU 激活函数）：开始训练")
print("=" * 70)
print("""
公式：y = w1 * ReLU(w0*x + b0) + b1
关键区别：ReLU(w0*x+b0) 在 w0*x+b0 < 0 时输出 0，
        在 >=0 时输出 w0*x+b0。
→ 这个『拐弯』动作，让它不再是直线！""")

w0, b0 = 1.0, 0.5
w1, b1 = 1.0, 0.5
lr = 0.05

def netB(x):
    return w1 * np.maximum(0, w0 * x + b0) + b1

print("\n【epoch 0】初始化 w0=1.0,b0=0.5,w1=1.0,b1=0.5")
for i in range(len(xs)):
    print(f"  输入 x={xs[i]:>4}  →  ReLU前={w0*xs[i]+b0:>6.3f} → ReLU后={np.maximum(0,w0*xs[i]+b0):>6.3f} → 输出 {netB(xs[i]):>6.3f}  (目标 {ys[i]:>4})")

print("\n训练 2000 轮...")

for epoch in range(2000):
    z = w0 * xs + b0            # 线性部分
    h = np.maximum(0, z)        # ReLU：负数→0
    pred = w1 * h + b1          # 输出
    loss = np.mean((pred - ys) ** 2)

    # 反向传播（ReLU的导数是：z>0时=1，z<=0时=0）
    d_loss_d_pred = 2 * (pred - ys) / len(xs)
    d_w1 = np.sum(d_loss_d_pred * h)
    d_b1 = np.sum(d_loss_d_pred)
    relu_grad = (z > 0).astype(float)          # ReLU 的导数
    d_w0 = np.sum(d_loss_d_pred * w1 * relu_grad * xs)
    d_b0 = np.sum(d_loss_d_pred * w1 * relu_grad)

    w0 -= lr * d_w0
    b0 -= lr * d_b0
    w1 -= lr * d_w1
    b1 -= lr * d_b1

    if epoch in [0, 1, 2, 100, 1000, 1999]:
        print(f"  epoch {epoch:<5} loss={loss:.5f}")

print("\n【网络B 训练完成后】的输出：")
for i in range(len(xs)):
    print(f"  输入 x={xs[i]:>4}  →  网络输出 {netB(xs[i]):>6.3f}   (目标 {ys[i]:>4})")

print("\n  它学会了在 x 的一个区间『拐弯』，所以能同时处理抛物线两端")
print("  → loss 可以降到非常低！")

# =========================================================
# 打印最终结论
# =========================================================
print("\n" + "=" * 70)
print("最终对比结论")
print("=" * 70)
print("""
网络A（无激活）：无论怎么训练，本质永远是 y=kx+c 一条直线
                → 永远学不会抛物线 → 这就是『没有激活函数的网络白搭』

网络B（有ReLU）：因为 ReLU 会『拐弯』，网络能弯出曲线
                → 学会了抛物线 → 这就是『激活函数注入非线性』

一句话：
   没有激活函数 = 只会画直线（再深也没用）
   有激活函数   = 能画任意曲线（越深越强）
""")

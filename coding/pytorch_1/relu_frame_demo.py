# -*- coding: utf-8 -*-
"""
【逐帧动画版】每一轮训练生成一张图，看 ReLU 折线如何"拐弯拼出曲线"。
16 个 ReLU 神经元去拟合抛物线 y = x²。
每 15 轮保存一张快照，最终拼成 GIF 动图。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = "relu_frames"
os.makedirs(OUT, exist_ok=True)

x = np.linspace(-3, 3, 500)
x_train = np.linspace(-3, 3, 200).reshape(1, -1)
y_target = x_train ** 2

np.random.seed(42)
M = 16
W1 = np.random.randn(M, 1) * 1.0
b1 = np.random.randn(M, 1) * 1.0
W2 = np.random.randn(1, M) * 0.1
b2 = np.zeros((1, 1))

def forward(x):
    h = np.maximum(0, W1 @ x + b1)
    y = W2 @ h + b2
    return h, y

frames = []
epochs_to_save = [0, 1, 3, 8, 20, 50, 120, 300, 800, 1500, 2500, 3999]
next_save = 0
lr = 0.01

for epoch in range(4000):
    h, y = forward(x_train)
    L = np.mean((y - y_target) ** 2)

    if next_save < len(epochs_to_save) and epoch == epochs_to_save[next_save]:
        # ===== 画这一帧 =====
        fig, ax = plt.subplots(figsize=(8.5, 6))
        # 目标曲线
        ax.plot(x, x**2, 'k--', lw=2, label='目标 y=x²')
        # 每个 ReLU 神经元的折线段（W2[i] 是它的权重）
        hx = np.maximum(0, W1 @ x.reshape(1, -1) + b1)   # (16, 500)
        for i in range(M):
            if abs(W2[0, i]) > 1e-3:
                ax.plot(x, W2[0, i] * hx[i], color='#9999ff', lw=0.8, alpha=0.5)
        # 网络最终输出（用画图的 x 重新算，保证点数一致）
        _, y_plot = forward(x.reshape(1, -1))
        ax.plot(x, y_plot.ravel(), 'r-', lw=3, label='网络输出 (16段折线之和)')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1, 10)
        ax.set_title(f"epoch {epoch}    loss = {L:.5f}", fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left')
        ax.axhline(0, color='gray', lw=0.5)
        fig.tight_layout()
        fname = f"{OUT}/frame_{epoch:04d}.png"
        fig.savefig(fname, dpi=95)
        plt.close(fig)
        frames.append(fname)
        print(f"已保存帧: {fname}  loss={L:.5f}")
        next_save += 1

    # 反向传播
    dy = 2 * (y - y_target) / y_target.size
    dW2 = dy @ h.T
    db2 = dy.sum(axis=1, keepdims=True)
    dh = W2.T @ dy
    mask = (h > 0).astype(float)
    dW1 = (dh * mask) @ x_train.T
    db1 = (dh * mask).sum(axis=1, keepdims=True)
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

# ===== 拼成 GIF =====
try:
    import imageio
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio"])
    import imageio

images = [imageio.imread(f) for f in frames]
imageio.mimsave('relu_animation.gif', images, duration=0.35)
print(f"\n动图已生成: relu_animation.gif  ({len(images)} 帧)")
print("逐帧图片保存在文件夹 relu_frames/ 里")

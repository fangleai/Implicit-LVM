import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def optimizer_step(loss, optimizer, retain_g=False):
    optimizer.zero_grad()
    loss.backward(retain_graph=retain_g)
    optimizer.step()


def create_scatter(z, save_path):
    fig = plt.figure(figsize=(4, 4), facecolor='w')
    ax = fig.add_axes([0, 0, 1, 1])

    m = z.shape[0] // 4
    for i in range(4):
        z_i = z[(i * m):((i + 1) * m), :]
        plt.scatter(z_i[:, 0], z_i[:, 1], edgecolor='none', alpha=0.5)

    ax.axis('off')
    fig.savefig(save_path)
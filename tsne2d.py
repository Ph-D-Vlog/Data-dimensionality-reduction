import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml


# 加载 MNIST 数据集
mnist = fetch_openml('mnist_784')
x = mnist.data / 255.0
y = mnist.target.astype(int)

# 使用 t-SNE 对数据进行降维，设置 n_jobs 为 -1 以使用所有可用的 CPU 核心
tsne = TSNE(n_components=2, n_jobs=-1)
x_tsne = tsne.fit_transform(x)

# 可视化降维后的数据，按类别用不同颜色
plt.figure(figsize=(10, 8))
for i in range(10):
    plt.scatter(x_tsne[y == i, 0], x_tsne[y == i, 1], s=1, label=str(i))
plt.title('t-SNE Visualization of MNIST Dataset')  # 可视化 MNIST 数据集的 t-SNE
plt.xlabel('t-SNE Component 1')  # t-SNE 成分 1
plt.ylabel('t-SNE Component 2')  # t-SNE 成分 2
plt.legend(title='Digit')  # 数字类别
plt.grid(True)  # 显示网格
plt.show()

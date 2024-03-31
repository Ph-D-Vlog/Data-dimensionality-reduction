from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


# 加载 MNIST 数据集
mnist = fetch_openml('mnist_784')
x = mnist.data / 255.0
y = mnist.target.astype(int)

# 使用 UMAP 进行降维
umap = UMAP(n_components=2, n_jobs=-1)
x_umap = umap.fit_transform(x)

# 可视化降维后的数据，按类别用不同颜色
plt.figure(figsize=(10, 8))
for i in range(10):
    plt.scatter(x_umap[y == i, 0], x_umap[y == i, 1], s=1, label=str(i))
plt.title('UMAP Visualization of the Dataset')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.legend(title='Digit')
plt.grid(True)
plt.show()

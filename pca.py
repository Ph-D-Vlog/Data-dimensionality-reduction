import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml


# 加载 MNIST 数据集
mnist = fetch_openml('mnist_784')
x = mnist.data / 255.0
y = mnist.target.astype(int)

# 使用 PCA 对数据进行降维
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

# 可视化降维后的数据，按类别用不同颜色
plt.figure(figsize=(10, 8))
for i in range(10):
    plt.scatter(x_pca[y == i, 0], x_pca[y == i, 1], s=1, label=str(i))
plt.title('PCA Visualization of MNIST Dataset')  # 可视化 MNIST 数据集的 PCA
plt.xlabel('Principal Component 1')  # 主成分 1
plt.ylabel('Principal Component 2')  # 主成分 2
plt.legend(title='Digit')  # 数字类别
plt.grid(True)  # 显示网格
plt.show()

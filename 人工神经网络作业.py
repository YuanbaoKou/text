import matplotlib.pyplot as plt#导入库函数
import numpy as np
#创建一个简单的表格
x = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x.shape)
y = np.square(x) - 0.5 + noise
plt.scatter(x, y)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)
from matplotlib.image import imread

image = imread('./images/mountain.jpg')
X = image.reshape(-1, 3)
kmeans = KMeans(n_clusters=3, random_state=42).fit(X)  # 改变n_clusters来改变分割色块数
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape) / 255

plt.imshow(image)
plt.axis('off')
plt.show()

plt.imshow(segmented_img)
plt.axis('off')
plt.show()

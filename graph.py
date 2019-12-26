from sklearn.manifold import TSNE
import pandas as pd

from get_tfidf_matrix import get_uci_data
from main import datasets1
data_zs,r,n_cluster=get_uci_data(5,datasets1)
print(data_zs)
tsne = TSNE()
tsne.fit_transform(data_zs)  # 进行数据降维,降成两维
# a=tsne.fit_transform(data_zs) #a是一个array,a相当于下面的tsne_embedding_
tsne = pd.DataFrame(tsne.embedding_)  # 转换数据格式, index=data_zs.index

colors=['r.','go','b*']

import matplotlib.pyplot as plt
for i in range(n_cluster):
    d = tsne[r == i]
    plt.plot(d[0], d[1], 'r.')

d = tsne[r == 1]
plt.plot(d[0], d[1], 'go')

d = tsne[r == 2]
plt.plot(d[0], d[1], 'b*')

plt.show()
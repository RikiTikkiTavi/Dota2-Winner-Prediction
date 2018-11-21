from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt

def visualize(data):
    data_embedded = TSNE(n_components=2).fit_transform(data)
    print(data_embedded)
    plt.plot(data_embedded)
    plt.show()

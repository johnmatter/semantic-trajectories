import umap
import matplotlib.pyplot as plt
import numpy as np

def plot_memory_space(memory_vectors, labels=None, title="Memory Space", n_components=2):
    """
    Visualize the compressed memory vectors in 2D or 3D using UMAP.
    """
    memory_array = np.array(list(memory_vectors.values()))
    
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    embedding = reducer.fit_transform(memory_array)

    if n_components == 2:
        plt.figure(figsize=(10, 7))
        plt.scatter(embedding[:, 0], embedding[:, 1], s=50, c='blue', alpha=0.6)

        if labels:
            for i, label in enumerate(labels):
                plt.text(embedding[i, 0], embedding[i, 1], label, fontsize=9)

        plt.title(title)
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.grid(True)
        plt.show()

    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], s=50, c='blue', alpha=0.6)

        if labels:
            for i, label in enumerate(labels):
                ax.text(embedding[i, 0], embedding[i, 1], embedding[i, 2], label, fontsize=8)

        ax.set_title(title)
        plt.show()

    else:
        raise ValueError("Only 2D or 3D supported for now.")

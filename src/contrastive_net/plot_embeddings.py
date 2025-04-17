import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import (
    EMBEDDINGS_PLOT_FILE_PATH,
    SKY_FINDER_SKY_CLASSES,
    EMBEDDINGS_FILE_PATH,
    PROJECTION_DIM,
    SEED,
)


def main() -> None:
    # Read file
    print("▶️  Reading embeddings file...")
    with open(EMBEDDINGS_FILE_PATH, "r") as f:
        embeddings_data = json.load(f)

    # Convert data to numpy array
    embeddings = np.zeros((len(embeddings_data), PROJECTION_DIM))
    embedding_sky_classes = np.zeros((len(embeddings_data), 1))
    for id, embedding_data in enumerate(embeddings_data.values()):
        sky_type = embedding_data["sky_type"]
        embedding = embedding_data["embedding"]
        embeddings[id] = np.array(embedding)
        embedding_sky_classes[id] = SKY_FINDER_SKY_CLASSES.index(sky_type)

    # Get TSNE and fit data
    print("▶️  Fitting T-SNE...")
    tsne = TSNE(
        metric="cosine",
        n_components=2,
        random_state=SEED,
        perplexity=100,
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot embeddings
    print("▶️  Plotting T-SNE...")
    plt.figure(figsize=(10, 10))
    colors = ["blue", "orange", "red"]
    cmap = mcolors.ListedColormap(colors[: len(SKY_FINDER_SKY_CLASSES)])
    plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=embedding_sky_classes,
        cmap=cmap,
        s=10,
        alpha=0.7,
    )
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=colors[i],
            markersize=5,
            label=sky_type,
        )
        for i, sky_type in enumerate(SKY_FINDER_SKY_CLASSES)
    ]
    plt.legend(handles=legend_elements, title="Sky Types")
    plt.title("T-SNE Visualization of Sky Type Embeddings")
    plt.xlabel("TSNE Component 1")
    plt.ylabel("TSNE Component 2")
    plt.grid(True, alpha=0.3)
    plt.savefig(EMBEDDINGS_PLOT_FILE_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print("✅ T-SNE plot saved as tsne_plot.png")


if __name__ == "__main__":
    main()

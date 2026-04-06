"""Basic t-SNE visualization for node embeddings."""

import torch
from torch import Tensor
import plotly.express as px
import numpy as np
from sklearn.manifold import TSNE

def plot_tsne_embeddings(node_emb: Tensor, labels: Tensor = None, save_path: str = 'tsne.png'):
    """Plot 2D t-SNE of node embeddings.

    Args:
        node_emb: [N, D] node embeddings
        labels: Optional [N] labels for coloring
        save_path: Save plot path
    """

    emb_np = node_emb.cpu().numpy()
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(emb_np)
    
    if labels is None:
        fig = px.scatter(x=emb_2d[:,0], y=emb_2d[:,1], title='Node Embeddings t-SNE')
    else:
        labels_np = labels.cpu().numpy()
        fig = px.scatter(x=emb_2d[:,0], y=emb_2d[:,1], color=labels_np.astype(str), 
                         title='Node Embeddings t-SNE (colored by label)')
    
    fig.write_html(save_path.replace('.png', '.html'))
    fig.write_image(save_path)
    print(f't-SNE saved to {save_path}')


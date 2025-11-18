from dataclasses import dataclass
from typing import List, Optional
import torch

@dataclass
class Config:
    # Model
    model_name: str = "Qwen/Qwen1.5-1.8B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Tokenization / generation
    max_length: int = 64

    # Data
    corpus: Optional[List[str]] = None

    # Graph building
    graph_mode: str = "knn"     # {"knn", "threshold"}
    knn_k: int = 8
    sim_threshold: float = 0.70
    use_cosine: bool = True

    # Anchors / LoT-style features (global)
    anchor_k: int = 16
    anchor_temp: float = 0.7

    # Clustering per layer
    cluster_method: str = "auto"   # {"auto","leiden","hdbscan","dbscan","kmeans"}
    n_clusters_kmeans: int = 6
    hdbscan_min_cluster_size: int = 4

    # DR / embeddings
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.05
    umap_metric: str = "cosine"
    use_global_3d_umap: bool = False

    # Pooling for UMAP fit
    fit_pool_per_layer: int = 512

    # Volume grid (MRI view)
    grid_res: int = 128
    kde_bandwidth: float = 0.15
    use_hist2d: bool = True

    # Output
    out_dir: str = "qwen_mri3d_outputs"
    plotly_html: str = "qwen_layers_3d.html"
    volume_npz: str = "qwen_density_volume.npz"
    volume_screenshot: str = "qwen_volume.png"

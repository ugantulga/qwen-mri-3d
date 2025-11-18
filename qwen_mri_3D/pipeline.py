import os
import math
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
from torch import nn

import networkx as nx
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Dimensionality reduction
import umap

# Neighbors & clustering
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

# Plotly for interactive 3D
import plotly.graph_objects as go

# Optional libs (only use if you have enough processing power)
try:
    import hdbscan  # Robust density-based clustering
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

try:
    import igraph as ig
    import leidenalg as la
    HAS_IGRAPH_LEIDEN = True
except Exception:
    HAS_IGRAPH_LEIDEN = False

try:
    import pyvista as pv  # Volume & isosurfaces (VTK)
    HAS_PYVISTA = True
except Exception:
    HAS_PYVISTA = False

from scipy.linalg import orthogonal_procrustes  # For optional per-layer orientation alignment


# ====== 1. Configuration ========
@dataclass
class Config:
    # Model
    model_name: str = "Qwen/Qwen1.5-1.8B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Tokenization / generation
    max_length: int = 64  # truncate inputs for speed/memory

    # Data
    corpus: List[str] = None  # set below
    # If None, uses DEFAULT_CORPUS defined below

    # Graph building
    graph_mode: str = "knn"  # {"knn", "threshold"}
    knn_k: int = 8           # neighbors per token (used if graph_mode="knn")
    sim_threshold: float = 0.70  # used if graph_mode="threshold"
    use_cosine: bool = True

    # Anchors / LoT-style features (global)
    anchor_k: int = 16        # number of global prototypes (KMeans on pooled states)
    anchor_temp: float = 0.7  # softmax temperature for converting distances to probs

    # Clustering per layer
    cluster_method: str = "auto"  # {"auto","leiden","hdbscan","dbscan","kmeans"}
    n_clusters_kmeans: int = 6    # fallback for kmeans
    hdbscan_min_cluster_size: int = 4

    # DR / embeddings
    umap_n_neighbors: int = 30
    umap_min_dist: float = 0.05
    umap_metric: str = "cosine"   # hidden states are directional → cosine works well
    use_global_3d_umap: bool = False  # if True, compute a single 3D manifold for all states

    # Pooling for UMAP fit
    fit_pool_per_layer: int = 512  # number of states sampled per layer to fit UMAP

    # Volume grid (MRI view)
    grid_res: int = 128  # voxel resolution in x/y; z = num_layers
    kde_bandwidth: float = 0.15   # KDE bandwidth in manifold space (if using KDE)
    use_hist2d: bool = True       # if True, use histogram2d instead of KDE for speed

    # Output
    out_dir: str = "qwen_mri3d_outputs"
    plotly_html: str = "qwen_layers_3d.html"
    volume_npz: str = "qwen_density_volume.npz"  # saved if PyVista isn't available
    volume_screenshot: str = "qwen_volume.png"   # if PyVista is available


# Default corpus (small and diverse; adjust freely)
DEFAULT_CORPUS = [
    "The cat sat on the mat and watched.",
    "Machine learning models process data using neural networks.",
    "Climate change affects ecosystems around the world.",
    "Quantum computers use superposition for parallel computation.",
    "The universe contains billions of galaxies.",
    "Artificial intelligence transforms how we work.",
    "DNA stores genetic information in cells.",
    "Ocean currents regulate Earth's climate system.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Blockchain technology enables decentralized systems."
]


# ====== 2. Utilities =============================================================================
def seed_everything(seed: int = 42):
    """Determinism for reproducibility in layouts/UMAP/kmeans."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity for rows of X."""
    # X: (N, D)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / norms
    return Xn @ Xn.T


def build_knn_graph(coords: np.ndarray, k: int, metric: str = "cosine") -> nx.Graph:
    """
    Build an undirected kNN graph for the points in coords.
    coords: (N, D)
    """
    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(coords)), metric=metric)  # +1 to include self
    nbrs.fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    G = nx.Graph()
    G.add_nodes_from(range(len(coords)))
    # Connect i to its top-k neighbors (skip index 0 which is itself)
    for i in range(len(coords)):
        for j in indices[i, 1:]:  # skip self
            G.add_edge(int(i), int(j))
    return G


def build_threshold_graph(H: np.ndarray, threshold: float, use_cosine: bool = True) -> nx.Graph:
    """
    Build graph by thresholding pairwise similarities in the original hidden-state space.
    H: (N, D) hidden states for a single layer
    """
    if use_cosine:
        S = cosine_similarity_matrix(H)
    else:
        S = H @ H.T  # dot product

    N = S.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i + 1, N):
            if S[i, j] > threshold:
                G.add_edge(i, j, weight=float(S[i, j]))
    return G


def percolation_stats(G: nx.Graph) -> Dict[str, float]:
    """
    Compute percolation observables (φ, #clusters, χ) as in your notebook.
    φ  : fraction of nodes in the Giant Connected Component (GCC)
    χ  : mean size of components excluding GCC
    """
    n = G.number_of_nodes()
    if n == 0:
        return dict(phi=0.0, num_clusters=0, chi=0.0, largest_component_size=0, component_sizes=[])

    comps = list(nx.connected_components(G))
    sizes = [len(c) for c in comps]
    if not sizes:
        return dict(phi=0.0, num_clusters=0, chi=0.0, largest_component_size=0, component_sizes=[])

    largest = max(sizes)
    phi = largest / n

    non_gcc_sizes = [s for s in sizes if s != largest]
    chi = float(np.mean(non_gcc_sizes)) if non_gcc_sizes else 0.0

    return dict(phi=float(phi),
                num_clusters=len(comps),
                chi=float(chi),
                largest_component_size=largest,
                component_sizes=sorted(sizes, reverse=True))


def leiden_communities(G: nx.Graph) -> np.ndarray:
    """
    Community detection using Leiden (igraph), if available.
    Returns an array of cluster ids for nodes 0..N-1.
    """
    if not HAS_IGRAPH_LEIDEN:
        raise RuntimeError("igraph+leidenalg not available")

    # Convert nx → igraph
    mapping = {n: i for i, n in enumerate(G.nodes())}
    edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
    ig_g = ig.Graph(n=len(mapping), edges=edges, directed=False)
    part = la.find_partition(ig_g, la.RBConfigurationVertexPartition)  # robust default
    labels = np.zeros(len(mapping), dtype=int)
    for cid, comm in enumerate(part):
        for node in comm:
            labels[node] = cid
    return labels


def cluster_layer(features: np.ndarray,
                  G: Optional[nx.Graph],
                  method: str,
                  n_clusters_kmeans: int = 6,
                  hdbscan_min_cluster_size: int = 4) -> np.ndarray:
    """
    Cluster layer states to get cluster labels.
      - If Leiden: requires G (graph) and igraph/leidenalg
      - If HDBSCAN: density-based clustering in feature space
      - If DBSCAN: fallback density-based (scikit-learn)
      - If KMeans: fallback centroid clustering
    """
    method = method.lower()
    N = len(features)

    if method == "auto":
        # Prefer Leiden (graph) → HDBSCAN → KMeans
        if HAS_IGRAPH_LEIDEN and G is not None and G.number_of_edges() > 0:
            return leiden_communities(G)
        elif HAS_HDBSCAN and N >= 5:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size,
                                        metric='euclidean')
            labels = clusterer.fit_predict(features)
            # HDBSCAN: -1 = noise. Keep as its own "noise" cluster id or remap
            return labels
        else:
            km = KMeans(n_clusters=min(n_clusters_kmeans, max(2, N // 3)),
                        n_init="auto", random_state=42)
            return km.fit_predict(features)

    if method == "leiden":
        if G is None or not HAS_IGRAPH_LEIDEN:
            raise RuntimeError("Leiden requires a graph and igraph+leidenalg.")
        return leiden_communities(G)

    if method == "hdbscan":
        if not HAS_HDBSCAN:
            raise RuntimeError("hdbscan not installed")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, metric='euclidean')
        return clusterer.fit_predict(features)

    if method == "dbscan":
        db = DBSCAN(eps=0.5, min_samples=4, metric='euclidean')
        return db.fit_predict(features)

    if method == "kmeans":
        km = KMeans(n_clusters=min(n_clusters_kmeans, max(2, N // 3)),
                    n_init="auto", random_state=42)
        return km.fit_predict(features)

    raise ValueError(f"Unknown cluster method: {method}")


def orthogonal_align(A_ref: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Align B to A_ref by an orthogonal rotation (Procrustes),
    preserving geometry but removing arbitrary orientation flips.
    """
    R, _ = orthogonal_procrustes(B - B.mean(0), A_ref - A_ref.mean(0))
    return (B - B.mean(0)) @ R + A_ref.mean(0)


def entropy_from_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy for each row; p is (N, K) with rows summing ~1."""
    return -np.sum(p * np.log(p + eps), axis=1)


# ====== 3. Model I/O (hidden states) =============================================================
@dataclass
class HiddenStatesBundle:
    """
    Encapsulates a single input's hidden states and metadata.
        hidden_layers: list of np.ndarray of shape (T, D), length = num_layers+1 (incl. embedding)
        tokens       : list of token strings of length T
    """
    hidden_layers: List[np.ndarray]
    tokens: List[str]


def load_qwen(model_name: str, device: str, dtype: torch.dtype):
    """
    Load Qwen with output_hidden_states=True. We use AutoTokenizer for broader compatibility.
    """
    print(f"[Load] {model_name} on {device} ({dtype})")
    config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    model.eval().to(device)
    if device == "cuda" and dtype == torch.float16:
        model = model.half()
    return model, tok


@torch.no_grad()
def extract_hidden_states(model, tokenizer, text: str, max_length: int, device: str) -> HiddenStatesBundle:
    """
    Run a single forward pass to collect all hidden states (incl. embedding layer).
    Returns CPU numpy arrays to keep GPU memory low.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    out = model(**inputs)
    # Tuple length = num_layers + 1 (embedding)
    hs = [h[0].detach().float().cpu().numpy() for h in out.hidden_states]  # shapes: (T, D)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return HiddenStatesBundle(hidden_layers=hs, tokens=tokens)


# ====== 4. LoT-style anchors & features ==========================================================
def fit_global_anchors(all_states_sampled: np.ndarray, K: int, random_state: int = 42) -> np.ndarray:
    """
    Fit KMeans cluster centroids on a pooled set of states (from many layers/texts).
    These centroids are "anchors" (LoT-like choices) to build low-dim features:
      f(state) = [dist(state, anchor_j)]_{j=1..K}
    """
    print(f"[Anchors] Fitting {K} global centroids on {len(all_states_sampled)} states ...")
    kmeans = KMeans(n_clusters=K, n_init="auto", random_state=random_state)
    kmeans.fit(all_states_sampled)
    return kmeans.cluster_centers_  # (K, D)


def anchor_features(H: np.ndarray, anchors: np.ndarray, temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For states H (N,D) and anchors A (K,D):
      - Compute Euclidean distances to each anchor → Dists (N,K)
      - Convert to soft probabilities with exp(-Dist/T), normalize row-wise → P (N,K)
      - Uncertainty = entropy(P) (cf. LoT Eq. (6))
      - Top-anchor argmin distance for "consistency"-style comparisons (cf. Eq. (5))
    Returns (Dists, P, entropy)
    """
    # Distances (N, K)
    dists = pairwise_distances(H, anchors, metric="euclidean")  # (N,K)
    # Soft assignments
    logits = -dists / max(temperature, 1e-6)
    # Stable softmax
    logits = logits - logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True) + 1e-12
    # Uncertainty (entropy)
    H_unc = entropy_from_probs(P)
    return dists, P, H_unc


# ====== 5. Dimensionality reduction / embeddings ================================================
def fit_umap_2d(pool: np.ndarray,
                n_neighbors: int = 30,
                min_dist: float = 0.05,
                metric: str = "cosine",
                random_state: int = 42) -> umap.UMAP:
    """
    Fit UMAP once on a diverse pool across layers to preserve orientation.
    Later layers call .transform() to embed into the SAME 2D space → "MRI stack".
    """
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                        metric=metric, random_state=random_state)
    reducer.fit(pool)
    return reducer


def fit_umap_3d(all_states: np.ndarray,
                n_neighbors: int = 30,
                min_dist: float = 0.05,
                metric: str = "cosine",
                random_state: int = 42) -> np.ndarray:
    """
    Fit a global 3D UMAP embedding for all states at once (alternative to slice stack).
    Returns coords_3d (N,3) for the concatenated states passed in.
    """
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist,
                        metric=metric, random_state=random_state)
    return reducer.fit_transform(all_states)


# ====== 6. Volume construction (MRI) ============================================================
def stack_density_volume(xy_by_layer: List[np.ndarray],
                         grid_res: int,
                         use_hist2d: bool = True,
                         kde_bandwidth: float = 0.15) -> np.ndarray:
    """
    Construct a 3D volume by estimating 2D density on the (x,y) manifold per layer (slice).
      - If use_hist2d: fast uniform binning into grid_res x grid_res
      - Else: KDE (slower but smoother)
    Returns volume of shape (grid_res, grid_res, L) where L = #layers.
    """
    L = len(xy_by_layer)
    vol = np.zeros((grid_res, grid_res, L), dtype=np.float32)

    # Determine global bounds across layers to keep axes consistent
    all_xy = np.vstack([xy for xy in xy_by_layer if len(xy) > 0]) if L > 0 else np.zeros((0, 2))
    if len(all_xy) == 0:
        return vol
    x_min, y_min = all_xy.min(axis=0)
    x_max, y_max = all_xy.max(axis=0)
    # Slight padding
    pad = 1e-6
    x_edges = np.linspace(x_min - pad, x_max + pad, grid_res + 1)
    y_edges = np.linspace(y_min - pad, y_max + pad, grid_res + 1)

    for l, XY in enumerate(xy_by_layer):
        if len(XY) == 0:
            continue

        if use_hist2d:
            H, _, _ = np.histogram2d(XY[:, 0], XY[:, 1], bins=[x_edges, y_edges], density=False)
            vol[:, :, l] = H.T  # histogram2d returns [x_bins, y_bins] → transpose to align
        else:
            kde = KernelDensity(bandwidth=kde_bandwidth, kernel="gaussian")
            kde.fit(XY)
            # Evaluate KDE on grid centers
            xs = 0.5 * (x_edges[:-1] + x_edges[1:])
            ys = 0.5 * (y_edges[:-1] + y_edges[1:])
            xx, yy = np.meshgrid(xs, ys, indexing='xy')
            grid_points = np.column_stack([xx.ravel(), yy.ravel()])
            log_dens = kde.score_samples(grid_points)
            dens = np.exp(log_dens).reshape(grid_res, grid_res)
            vol[:, :, l] = dens

    # Normalize volume to [0,1] for rendering convenience
    if vol.max() > 0:
        vol = vol / vol.max()
    return vol


def render_volume_with_pyvista(volume: np.ndarray,
                               out_png: str,
                               opacity="sigmoid") -> None:
    """
    Visualize the 3D volume using PyVista/VTK (if installed); save a screenshot.
    """
    if not HAS_PYVISTA:
        raise RuntimeError("PyVista is not installed; cannot render volume.")
    pl = pv.Plotter()
    # Wrap NumPy array as a VTK image data; PyVista expects z as the 3rd axis
    vol_vtk = pv.wrap(volume)
    pl.add_volume(vol_vtk, opacity=opacity, shade=True)
    pl.show(screenshot=out_png)  # headless environments will still save a screenshot (if offscreen support)


# ====== 7. 3D Plotly visualization ==============================================================
def plotly_3d_layers(xy_layers: List[np.ndarray],
                     layer_tokens: List[List[str]],
                     layer_cluster_labels: List[np.ndarray],
                     layer_uncertainty: List[np.ndarray],
                     layer_graphs: List[nx.Graph],
                     connect_token_trajectories: bool = True,
                     title: str = "Qwen: 3D Cluster Formation (UMAP2D + Layer as Z)") -> go.Figure:
    """
    Build an interactive 3D Plotly figure:
      - Nodes per layer at (x, y, z=layer)
      - Edge segments (kNN or threshold graph) per layer
      - Trajectory lines: connect same token index across consecutive layers (optional)
      - Color nodes by cluster label; hover shows token & uncertainty
    """
    fig_data = []

    # Build a color per layer node trace
    for l, (xy, tokens, labels, unc, G) in enumerate(zip(xy_layers, layer_tokens, layer_cluster_labels, layer_uncertainty, layer_graphs)):
        if len(xy) == 0:
            continue
        x, y = xy[:, 0], xy[:, 1]
        z = np.full_like(x, l, dtype=float)

        # --- Nodes
        node_text = [f"layer={l} | idx={i}<br>token={tokens[i]}<br>cluster={int(labels[i])}<br>uncertainty={unc[i]:.3f}"
                     for i in range(len(tokens))]
        node_trace = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            name=f"Layer {l}",
            marker=dict(
                size=4,
                opacity=0.7,
                color=labels,  # cluster ID → color scale
                colorscale='Viridis',
                showscale=(l == 0)  # show scale once
            ),
            text=node_text,
            hovertemplate="%{text}<extra></extra>"
        )
        fig_data.append(node_trace)

        # --- Intra-layer edges (kNN or threshold)
        if G is not None and G.number_of_edges() > 0:
            edge_x, edge_y, edge_z = [], [], []
            for u, v in G.edges():
                edge_x += [x[u], x[v], None]
                edge_y += [y[u], y[v], None]
                edge_z += [z[u], z[v], None]
            edge_trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(width=1),
                opacity=0.15,
                name=f"Edges L{l}"
            )
            fig_data.append(edge_trace)

    # --- Trajectories: connect same token index across layers
    if connect_token_trajectories:
        # Only meaningful if tokenization length T is constant across layers (it is)
        # We'll draw faint polylines for each position i across l=0..L-1
        L = len(xy_layers)
        if L > 1:
            T = min(len(xy_layers[l]) for l in range(L))
            for i in range(T):
                xs = [xy_layers[l][i, 0] for l in range(L)]
                ys = [xy_layers[l][i, 1] for l in range(L)]
                zs = list(range(L))
                traj = go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode='lines',
                    line=dict(width=1),
                    opacity=0.15,
                    name=f"traj_{i}",
                    hoverinfo='skip'
                )
                fig_data.append(traj)

    fig = go.Figure(data=fig_data)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="UMAP X",
            yaxis_title="UMAP Y",
            zaxis_title="Layer (depth)"
        ),
        height=900,
        showlegend=False
    )
    return fig


# ====== 8. Orchestration ========================================================================
def run_pipeline(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)
    seed_everything(42)

    # 8.1 Load model
    model, tok = load_qwen(cfg.model_name, cfg.device, cfg.dtype)

    # 8.2 Collect hidden states for one representative text (detailed viz) + for pool
    #     You can extend to many texts; we keep a single text for clarity & speed.
    texts = cfg.corpus or DEFAULT_CORPUS
    main_text = texts[2]  # pick one to visualize in detail
    print(f"[Input] Example text: {main_text!r}")

    # Hidden states for main text
    main_bundle = extract_hidden_states(model, tok, main_text, cfg.max_length, cfg.device)
    layers_np: List[np.ndarray] = main_bundle.hidden_layers   # list of (T,D), length L_all = num_layers+1
    tokens = main_bundle.tokens                               # list of length T
    L_all = len(layers_np)
    print(f"[Hidden] Layers (incl. embedding): {L_all}, Tokens: {len(tokens)}")

    # 8.3 Build a pool of states (across a few texts & layers) to fit anchors + UMAP
    pool_states = []
    # Sample across first few texts to improve diversity (lightweight)
    for t in texts[: min(5, len(texts))]:
        b = extract_hidden_states(model, tok, t, cfg.max_length, cfg.device)
        # Take a subset from each layer to limit pool size
        for H in b.hidden_layers:
            T = len(H)
            take = min(cfg.fit_pool_per_layer, T)
            idx = np.random.choice(T, size=take, replace=False)
            pool_states.append(H[idx])
    pool_states = np.vstack(pool_states) if len(pool_states) else layers_np[-1]
    print(f"[Pool] Pooled states for anchors/UMAP: {pool_states.shape}")

    # 8.4 Fit global anchors (LoT-style features)
    anchors = fit_global_anchors(pool_states, cfg.anchor_k)
    # Save anchors for reproducibility
    np.save(os.path.join(cfg.out_dir, "anchors.npy"), anchors)

    # 8.5 Build per-layer features for main text (LoT-style distances & uncertainty)
    layer_features = []      # list of (T,K)
    layer_uncertainties = [] # list of (T,)
    layer_top_anchor = []    # list of (T,) argmin-id

    for l, H in enumerate(layers_np):
        dists, P, H_unc = anchor_features(H, anchors, cfg.anchor_temp)
        layer_features.append(dists)              # N x K distances (lower = closer)
        layer_uncertainties.append(H_unc)         # N
        layer_top_anchor.append(np.argmin(dists, axis=1))  # closest anchor id per token

    # 8.6 Consistency metric (LoT Eq. (5)): does layer's top anchor match final layer's?
    final_top = layer_top_anchor[-1]
    layer_consistency = []
    for l in range(L_all):
        cons = (layer_top_anchor[l] == final_top).astype(np.int32)  # 1 if matches, 0 otherwise
        layer_consistency.append(cons)

    # 8.7 Build per-layer graphs (kNN by default) on FEATURE space for stability
    layer_graphs = []
    for l in range(L_all):
        feats = layer_features[l]
        if cfg.graph_mode == "knn":
            G = build_knn_graph(feats, cfg.knn_k, metric="euclidean")  # kNN in feature space
        else:
            # Threshold graph in original hidden space (as in your notebook)
            G = build_threshold_graph(layers_np[l], cfg.sim_threshold, use_cosine=cfg.use_cosine)
        layer_graphs.append(G)

    # 8.8 Cluster per layer
    layer_cluster_labels = []
    for l in range(L_all):
        feats = layer_features[l]
        labels = cluster_layer(
            feats,
            layer_graphs[l],
            method=cfg.cluster_method,
            n_clusters_kmeans=cfg.n_clusters_kmeans,
            hdbscan_min_cluster_size=cfg.hdbscan_min_cluster_size
        )
        layer_cluster_labels.append(labels)

    # 8.9 Percolation statistics (φ, #clusters, χ) per layer (as in your notebook)
    percolation = []
    for l in range(L_all):
        stats = percolation_stats(layer_graphs[l])
        percolation.append(stats)
    # Save percolation series
    with open(os.path.join(cfg.out_dir, "percolation_stats.json"), "w") as f:
        json.dump(percolation, f, indent=2)
    print(f"[Percolation] Saved per-layer stats → percolation_stats.json")

    # 8.10 Common 2D manifold via UMAP (fit-once on the pool), then transform each layer
    reducer2d = fit_umap_2d(pool_states,
                            n_neighbors=cfg.umap_n_neighbors,
                            min_dist=cfg.umap_min_dist,
                            metric=cfg.umap_metric)
    xy_by_layer = [reducer2d.transform(layers_np[l]) for l in range(L_all)]

    # OPTIONAL: orthogonal alignment across layers (helps if UMAP.transform still drifts)
    # for l in range(1, L_all):
    #     xy_by_layer[l] = orthogonal_align(xy_by_layer[l-1], xy_by_layer[l])

    # 8.11 Plotly 3D point+graph view: X,Y from UMAP; Z = layer index
    fig = plotly_3d_layers(
        xy_layers=xy_by_layer,
        layer_tokens=[tokens for _ in range(L_all)],
        layer_cluster_labels=layer_cluster_labels,
        layer_uncertainty=layer_uncertainties,
        layer_graphs=layer_graphs,
        connect_token_trajectories=True,
        title="Qwen: 3D Cluster Formation (UMAP2D + Layer as Z, LoT metrics on hover)"
    )
    html_path = os.path.join(cfg.out_dir, cfg.plotly_html)
    fig.write_html(html_path)
    print(f"[Plotly] 3D HTML saved → {html_path}")

    # 8.12 Build a true 3D volume (MRI): stack 2D densities across layers
    vol = stack_density_volume(xy_by_layer,
                               grid_res=cfg.grid_res,
                               use_hist2d=cfg.use_hist2d,
                               kde_bandwidth=cfg.kde_bandwidth)
    np.savez_compressed(os.path.join(cfg.out_dir, cfg.volume_npz), volume=vol)
    print(f"[Volume] Density volume saved (npz) → {os.path.join(cfg.out_dir, cfg.volume_npz)}")

    if HAS_PYVISTA:
        try:
            png_path = os.path.join(cfg.out_dir, cfg.volume_screenshot)
            render_volume_with_pyvista(vol, png_path, opacity="sigmoid")
            print(f"[PyVista] Volume screenshot saved → {png_path}")
        except Exception as e:
            warnings.warn(f"PyVista render failed: {e}. Volume npz is still saved.")
    else:
        print("[PyVista] Not installed. Skipping volume rendering. Use the saved .npz to render later.")

    # 8.13 Also write a small CSV with LoT metrics per layer for downstream plotting
    rows = []
    for l in range(L_all):
        for i in range(len(tokens)):
            rows.append(dict(
                layer=l,
                idx=i,
                token=tokens[i],
                cluster=int(layer_cluster_labels[l][i]),
                uncertainty=float(layer_uncertainties[l][i]),
                consistency=int(layer_consistency[l][i])
            ))
    df = pd.DataFrame(rows)
    csv_path = os.path.join(cfg.out_dir, "lot_metrics_per_token.csv")
    df.to_csv(csv_path, index=False)
    print(f"[LoT] Per-token metrics (uncertainty/consistency) saved → {csv_path}")

    # 8.14 Print a small textual summary of percolation observables
    print("\n=== PER-LAYER PERCOLATION SUMMARY (φ, #clusters, χ) ===")
    for l, st in enumerate(percolation):
        print(f"L={l:02d} | φ={st['phi']:.3f} | #C={st['num_clusters']:>3} | χ={st['chi']:.2f}")

    print("\nDONE. Open the Plotly HTML for the 3D scatter/graph.\n"
          "The volume .npz can be rendered later (e.g., load with NumPy/PyVista).")
  return cfg.out_dir


# ====== 9. Main =================================================================================
if __name__ == "__main__":
    cfg = Config()
    cfg.graph_mode = "threshold"
    cfg.sim_threshold = 0.70
    cfg.knn_k = 8
    cfg.anchor_k = 16
    cfg.cluster_method = "auto"  # or "leiden"/"hdbscan"/"dbscan"/"kmeans"
    cfg.use_global_3d_umap = False
    cfg.grid_res = 128

    run_pipeline(cfg)

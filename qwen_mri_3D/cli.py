import argparse
from .config import Config
from .pipeline import run_pipeline

def main():
    p = argparse.ArgumentParser(description="MRI-style 3D visualization of Qwen layers")
    p.add_argument("--model", dest="model_name", default=Config.model_name)
    p.add_argument("--device", default=None, help="cuda | cpu | auto (default: autodetect)")
    p.add_argument("--max-length", type=int, default=Config.max_length)
    p.add_argument("--graph-mode", choices=["knn", "threshold"], default=Config.graph_mode)
    p.add_argument("--knn-k", type=int, default=Config.knn_k)
    p.add_argument("--sim-threshold", type=float, default=Config.sim_threshold)
    p.add_argument("--anchor-k", type=int, default=Config.anchor_k)
    p.add_argument("--cluster-method",
                   choices=["auto","leiden","hdbscan","dbscan","kmeans"],
                   default=Config.cluster_method)
    p.add_argument("--umap-n-neighbors", type=int, default=Config.umap_n_neighbors)
    p.add_argument("--umap-min-dist", type=float, default=Config.umap_min_dist)
    p.add_argument("--umap-metric", default=Config.umap_metric)
    p.add_argument("--grid-res", type=int, default=Config.grid_res)
    p.add_argument("--out-dir", default=Config.out_dir)
    p.add_argument("--corpus", nargs="*", help="Override default corpus with your own sentences")
    args = p.parse_args()

    cfg = Config(
        model_name=args.model_name,
        device=(args.device if args.device not in (None, "auto") else Config().device),
        max_length=args.max_length,
        graph_mode=args.graph_mode,
        knn_k=args.knn_k,
        sim_threshold=args.sim_threshold,
        anchor_k=args.anchor_k,
        cluster_method=args.cluster_method,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        umap_metric=args.umap_metric,
        grid_res=args.grid_res,
        out_dir=args.out_dir,
        corpus=args.corpus if args.corpus else None,
    )

    out = run_pipeline(cfg)
    print(f"\nArtifacts written to: {out}\n")

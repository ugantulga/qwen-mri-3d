# 🧠 QWEN-MRI: 3D MRI-Style Visualization of Cluster Formation in Qwen Layers

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)]()

This repository visualizes how **token representations self-organize into clusters** across layers of the [Qwen](https://huggingface.co/Qwen) language model — inspired by *percolation theory* and the *Landscape of Thoughts* (LoT) framework.

We treat layer depth as the **z-axis**, UMAP2D embeddings as **(x,y)**, and use:
- ✅ kNN graphs & community detection (Leiden/HDBSCAN)
- ✅ LoT-style state features: `uncertainty`, `consistency`, `anchor distances`
- ✅ Interactive 3D Plotly view (tokens, edges, trajectories)
- ✅ Volumetric density rendering (PyVista) — like an "MRI scan" of representation space

![Teaser](examples/teaser_screenshot.png)  <!-- add later -->

> 🔬 Useful for probing *emergent modularity*, *phase transitions* in attention, or *faithfulness* of internal representations.

---

## 🔧 Installation

```bash
git clone https://github.com/ugantulga/qwen-mri-3d.git
cd qwen-mri-3d

# Core + Plotly/UMAP
pip install -r requirements.txt

# Optional: high-fidelity clustering & volume rendering
pip install hdbscan python-igraph leidenalg pyvista pyvistaqt

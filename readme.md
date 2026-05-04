# CAML: Constraint-Aligned loss with Manifold Lifting

Code for **CAML: Constraint-Aligned loss with Manifold Lifting**, presented at the _43rd International Conference on Machine Learning (ICML 2026)_.

## Getting Started

We implement the CAML based on PyTorch. Apart from this, there are no other specific environmental dependencies.
To reproduce the data in the paper, you can run `eval_heat.py`, `eval_poisson.py`, `eval_ns.py`, and `eval_helm.py`
for the Heat, Poisson, Navier-Stokes, and Helmholtz benchmarks, respectively. Hyperparameter settings can be found in
Appendix L.2. We will open source this repository in the camera-ready version.

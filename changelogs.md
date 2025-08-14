# CHANGELOG
The `changelogs.md` file is to track all changes related to `TAVIE`.

---

## Notable Addendum in v1.0.0
- **TAVIE-v1.0.0** comprises the implementation of the tangent transformation for SSG likelihoods as proposed in Roy, S., Dey, P., Pati, D., & Mallick, B. K. (2025), *A Generalized Tangent Approximation Framework for Strongly Super‑Gaussian Likelihoods*, [arXiv:2504.05431](https://arxiv.org/abs/2504.05431).

- TAVIE comparisons with a suite of competing *black-box variational inference* (BBVI) algorithms like DADVI, ADVI (MF), ADVI (FR) as well as MC algorithm NUTS from PyMC across all the SSG likelihoods.

- Application of TAVIE to U.S. Census 2000 data (performing Bayesian quantile regression) and STARmap dataset (performing Negative-Binomial regression for predicting spatially varying gene expressions).

---

## Folder and file information in v1.0.0
- `CompetingMethods`: contains all `Python` implementations of competing methods; `ADVI (MF)`, `ADVI (FR)`, and `MFVI`; across various SSG likelihood families.

- `dadvi`: includes all scripts and dependencies for running the `DADVI` algorithm. See [https://github.com/martiningram/dadvi](https://github.com/martiningram/dadvi) for additional details.

- `data`: stores datasets used in `TAVIE` applications, including the *U.S. 2000 Census* and *STARmap*.

- `results_compete`: contains results from `TAVIE` and competing methods across different SSG families, with varying sample sizes $n$ and features $p$.

- `results_data_study`: holds results from applying `TAVIE` and competing methods to real datasets.

- `TAVIE`: source code for the *TAVIE algorithm*, including `Python` classes for handling different SSG likelihoods.

- `census_QR_data_study_main.ipynb`: Jupyter Notebook demonstrating the complete workflow for `TAVIE` and competing methods in *quantile regression* on the *U.S. 2000 Census* dataset.

- `convergence.ipynb`: checks convergence of the competing BBVI methods against `TAVIE` using the *ELBO* history.

- `Laplace_comparisons_dadvi_tavie_bbvi.ipynb`: evaluates the performance of `TAVIE` against competing methods; `DADVI`, `ADVI (MF)`, `ADVI (FR)`, and `NUTS`; for the *Laplace* SSG likelihood.

- `NegBin_comparisons_dadvi_tavie_bbvi.ipynb`: same comparative analysis as above, applied to the *Negative-Binomial* SSG likelihood.

- `Student_comparisons_dadvi_tavie_mfvi_bbvi.ipynb`: same comparative analysis as above, applied to the *Student's-t* SSG likelihood.

- `STARmap_data_study.ipynb`: Jupyter Notebook demonstrating the complete workflow for `TAVIE` and competing methods in *Negative-Binomial regression* on the *STARmap* dataset for predicting gene expressions across different spatial locations.

- `TAVIE_examples.ipynb`: complete usage guide for running different `TAVIE` classes under various SSG likelihoods, demonstrating available functionalities in each class.

- `viewing_results.ipynb`: utility notebook for viewing additional comparison results and metrics for varying $n$ and $p$ across `TAVIE` and competing methods for different SSG families.

- `assets`: collection of plots, logos, and graphics used throughout the `TAVIE` package.

- `requirements.txt`: lists all the required `Python` packages to be installed and loaded for running `TAVIE`.

---
# TAVIE: A Generalized Tangent Approximation Framework for Strongly Super-Gaussian Likelihoods

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2504.05431-b31b1b)](https://arxiv.org/abs/2504.05431)
[![TAVIE JupyterBook](https://img.shields.io/badge/JupyterBook-view-blueviolet?logo=Jupyter)](https://roy-sr-007.github.io/TAVIE_JupyterBook)


This repository holds the implementations and source code of *Tangent Approximation based Variational InferencE* (**TAVIE**) proposed in Roy, S., Dey, P., Pati, D., & Mallick, B. K. (2025), *A Generalized Tangent Approximation Framework for Strongly Super‑Gaussian Likelihoods*, [arXiv:2504.05431](https://arxiv.org/abs/2504.05431).

### NEWS

- This is the first official release of TAVIE v1.0.0 on GitHub.
- Explore different example cases, settings and usage of TAVIE across different strongly super-Gaussian (SSG) likelihoods with comparison against other variational inference (VI) algorithms here in this *Jupyter Book* [![TAVIE JupyterBook](https://img.shields.io/badge/JupyterBook-view-blueviolet?logo=Jupyter)](https://roy-sr-007.github.io/TAVIE_JupyterBook).
- Application of TAVIE to real world data, to be added soon.

### Overview

*Variational inference* (VI), a concept rooted from statistical physics, has gained recent traction as a contender to prevalent Markov chain Monte Carlo (MCMC) sampling techniques used for posterior inference. VI has transformed approximate Bayesian inference through its power of scaling compute time under big data with applications extending out to the realm of *machine learning*, specifically in *graphical models* (Wainwright and Jordan, 2008; Jordan et al., 1999), *hidden Markov models* (HMMs) (MacKay, 1997), *latent class models* (Blei et al., 2003), and *neural networks* (NNs) (Graves, 2011). *Tangent approximation* (Jaakkola and Jordan, 2000), forming a popular class of VI techniques in intractable non-conjugate models has been used in diverse modeling frameworks like *low-rank approximations* (Srebro and Jaakkola, 2003), *sparse kernel machines* (Shi and Yu, 2019), and *online prediction* (Konagayoshi and Watanabe, 2019). However, these applications have been primarily confined to logistic regression setting with Ghosh et al., 2022 being the first ones to investigate optimality and algorithmic stability of tangent transformation based variational inference in logit and multinomial logit models. Taking a step forward, we develop TAVIE for SSG likelihood functions which encompasses a broad class of flexible probability models beyond the framework of logit models. Based on the principle of *convex duality*, TAVIE obtains a quadratic lower bound of the corresponding log-likelihood, thus inducing conjugacy with Gaussian priors over the model parameters. With TAVIE, we also address the challenge of rigorously characterizing the statistical behavior of the variational posterior by developing optimality (near-minimax variational risk bounds) under the fractional likelihood setup.

TAVIE works for a large class of strongly super-Gaussian (SSG) likelihoods, currently comprising of two types of families:
- *Type I families*: These comprise of linear regression models with heavy-tailed error distributions. Notable families of error distributions which can be addressed by TAVIE include the Laplace (Double-Exponential) and Student's-$t$. In general, any scaled-mixture of zero-mean Gaussian distributions have the SSG form, and thus can be implemented using TAVIE.
- *Type II families*: These comprise of generalized linear models (GLMs) with Binomial (including Bernoulli/Logistic) and Negative-Binomial response distributions.
- *Bayesian Quantlile Regression*: As an extension of the Type-I likelihood to the asymmetric Laplace distribution.

Due to the large class of models which can be fitted and infered from using TAVIE, it can be applied to a broad class of real world problems spanning applications in finance and economics, as well as in biostatistics viz., gene expression modeling, microbiome studies and neuroscience.

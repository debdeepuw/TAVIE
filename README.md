# TAVIE: A Generalized Tangent Approximation Framework for Strongly Super-Gaussian Likelihoods

![Python](https://img.shields.io/badge/Python-3.9-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![arXiv](https://img.shields.io/badge/arXiv-2504.05431-b31b1b)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)

This repository holds the implementations and source code of Tangent Approximation based Variational InferencE (TAVIE) proposed in Roy, S., Dey, P., Pati, D., & Mallick, B. K. (2025), *A Generalized Tangent Approximation Framework for Strongly Super‑Gaussian Likelihoods*, [arXiv:2504.05431](https://arxiv.org/abs/2504.05431).

TAVIE works for a large class of strongly super-Gaussian (SSG) likelihoods, currently comprising of two types of families:
- *Type I families*: These comprise of linear regression models with heavy-tailed error distributions. Notable families of error distributions which can be addressed by TAVIE include the Laplace (Double-Exponential) and Student's-$t$ families. In general, scaled-mixture of zero-mean Gaussian distributions have the SSG form, and thus can be addressed by TAVIE.
- *Type II families*: These comprise of generalized linear models with Binomial (including Bernoulli) and Negative-Binomial response distributions.
- *Bayesian Quantlile Regression*: As an extension of the Type-I likelihood to the asymmetric Laplace distribution.

Due to the large class of models which can be fitted and infered from using TAVIE, it can be applied to a broad class of real world applications including finance and economics, as well as in biostatistical applications like gene expression modeling, microbiome studies and neuroscience. Although the current release contains only simulated examples and comparisons with state-of-the-art variational algorithms, real world data studies will be uploaded soon.

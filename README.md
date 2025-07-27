# TAVIE: A Generalized Tangent Approximation Framework for Strongly Super-Gaussian Likelihoods

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2504.05431-b31b1b)](https://arxiv.org/abs/2504.05431)
[![Forks](https://img.shields.io/github/forks/Roy-SR-007/TAVIE)](https://github.com/Roy-SR-007/TAVIE/network)
[![Repo Size](https://img.shields.io/github/repo-size/Roy-SR-007/TAVIE)](https://github.com/Roy-SR-007/TAVIE)
[![Last Commit](https://img.shields.io/github/last-commit/Roy-SR-007/TAVIE)](https://github.com/Roy-SR-007/TAVIE/commits/main)
[![Issues](https://img.shields.io/github/issues/Roy-SR-007/TAVIE)](https://github.com/Roy-SR-007/TAVIE/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/Roy-SR-007/TAVIE)](https://github.com/Roy-SR-007/TAVIE/pulls)



<p align="center">
  <img src="assets/TAVIE_animated_logo.gif" alt="TAVIE_logo" width="650"/>
</p>

This repository holds the implementations and source code of *Tangent Approximation based Variational InferencE* (**TAVIE**) proposed in Roy, S., Dey, P., Pati, D., & Mallick, B. K. (2025), *A Generalized Tangent Approximation Framework for Strongly Super‑Gaussian Likelihoods*, [arXiv:2504.05431](https://arxiv.org/abs/2504.05431).

---

### Developers and Maintainers

**Somjit Roy**  
Department of Statistics  
Texas A&M University, College Station, TX, USA  

📧 Email: [sroy_123@tamu.edu](mailto:sroy_123@tamu.edu)  
🌐 Website: [https://roy-sr-007.github.io](https://roy-sr-007.github.io)

**Pritam Dey**  
Department of Statistics  
Texas A&M University, College Station, TX, USA  

📧 Email: [pritam.dey@tamu.edu](mailto:pritam.dey@tamu.edu)  
🌐 Website: [https://pritamdey.github.io](https://pritamdey.github.io)

---

### NEWS

- This is the first official release of TAVIE v1.0.0 on GitHub.
- Explore different example cases, settings and usage of TAVIE across various strongly super-Gaussian (SSG) likelihoods with comparison against other variational inference (VI) algorithms.
- Application of TAVIE to real world data, to be added soon.

---

### Overview

*Variational inference* (VI), a concept rooted from statistical physics, has gained recent traction as a contender to prevalent Markov chain Monte Carlo (MCMC) sampling techniques used for posterior inference. VI has transformed approximate Bayesian inference through its power of scaling compute time under big data with applications extending out to the realm of *machine learning*, specifically in *graphical models* ([Wainwright and Jordan, 2008](https://www.nowpublishers.com/article/Details/MAL-001); [Jordan et al., 1999](https://link.springer.com/article/10.1023/A:1007665907178)), *hidden Markov models* (HMMs) ([MacKay, 1997](http://www.inference.org.uk/mackay/ensemblePaper.pdf)), *latent class models* ([Blei et al., 2003](https://jmlr.csail.mit.edu/papers/v3/blei03a.html)), and *neural networks* (NNs) ([Graves, 2011](https://papers.nips.cc/paper_files/paper/2011/hash/7eb3c8be3d411e8ebfab08eba5f49632-Abstract.html)). *Tangent approximation* ([Jaakkola and Jordan, 2000](https://link.springer.com/article/10.1023/A:1008932416310)), forming a popular class of VI techniques in intractable non-conjugate models has been used in diverse modeling frameworks like *low-rank approximations* ([Srebro and Jaakkola, 2003](https://people.csail.mit.edu/tommi/papers/SreJaa-icml03.pdf)), *sparse kernel machines* ([Shi and Yu, 2019](https://proceedings.neurips.cc/paper/2019/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html)), and *online prediction* ([Konagayoshi and Watanabe, 2019](https://proceedings.mlr.press/v101/konagayoshi19a.html)). However, these applications have been primarily confined to logistic regression setting with [Ghosh et al., 2022](https://www.jmlr.org/papers/v23/21-0190.html) being the first ones to investigate optimality and algorithmic stability of tangent transformation based variational inference in logit and multinomial logit models. Taking a step forward, we develop TAVIE for SSG likelihood functions which encompasses a broad class of flexible probability models beyond the framework of logit models. Based on the principle of *convex duality*, TAVIE obtains a quadratic lower bound of the corresponding log-likelihood, thus inducing conjugacy with Gaussian priors over the model parameters. With TAVIE, we also address the challenge of rigorously characterizing the statistical behavior of the variational posterior by developing optimality (near-minimax variational risk bounds) under the fractional likelihood setup.

<p align="center">
  <img src="assets/tangent_bounds.gif" alt="Tangent Bound Animation" width="600"/>
  <br><em>Tangent minorizers for Student's-t likelihood, animated over 50 values of the variational parameter ξ</em>
</p>

TAVIE works for a large class of SSG likelihoods, currently comprising mainly of two types of families:
- *Type I families*: These comprise of linear regression models with heavy-tailed error distributions. Notable families of error distributions which can be addressed by TAVIE include the Laplace (Double-Exponential) and Student's-t. In general, any scaled-mixture of zero-mean Gaussian distributions have the SSG form, and thus can be implemented using TAVIE.
- *Type II families*: These comprise of generalized linear models (GLMs) with Binomial (including Bernoulli/Logistic) and Negative-Binomial response distributions.
- *Bayesian Quantlile Regression*: As an extension of the Type-I likelihood to the asymmetric Laplace distribution.

Due to the large class of models which can be fitted and infered from using TAVIE, it can be applied to a broad class of real world problems spanning applications in finance and economics, as well as in biostatistics viz., gene expression modeling, microbiome studies and neuroscience.

---

### Installation and Dependencies

To get started with TAVIE, which is build on `Python==3.13.5`, clone the current Github repository and install the required dependencies:

* `ipython==8.30.0`
* `matplotlib==3.10.3`
* `numpy==2.3.2`
* `pandas==2.3.1`
* `rich==14.1.0`
* `scikit_learn==1.7.1`
* `scipy==1.16.0`
* `torch==2.7.1`
* `tqdm==4.67.1`

```bash
# using SSH on bash
git clone git@github.com:Roy-SR-007/TAVIE.git

# or, using HTTPS on bash
git clone https://github.com/Roy-SR-007/TAVIE.git

# moving to the TAVIE directory
cd TAVIE

# install the required dependencies
pip install -r requirements.txt
```
---

### Functionalities of the TAVIE class

The `TAVIE` package wrapped in the TAVIE folder deals with the implementation of the corresponding VI algorithm under various SSG probability models. It provides flexible, modular support across different likelihood families and modeling tasks.


| Class             | Target SSG Model               | Supported Likelihoods                               | Prior Type                  |
|-------------------|--------------------------------|-----------------------------------------------------|-----------------------------|
| `TAVIE_loc_scale` | Location-scale family (Type I) | Laplace, Student’s-t, Custom location-scale         | Gaussian × Inverse-Gamma    |
| `TAVIE_QR`         | Quantile Regression           | Asymmetric Laplace (Quantile Regression)            | Gaussian                    |
| `TAVIE_type_II`    | GLMs (Exponential Family)     | Binomial, Negative Binomial                         | Gaussian                    |

---

For importing all these above-mentioned classes and initializing them at once:

```python
# importing all the TAVIE classes
from TAVIE import *

# initializing the TAVIE location-scale, TAVIE_QR, and TAVIE_type_II models respectively
## following is the location-scale example for 'laplace', other options are 'student' and 'loc_scale' (for general location-scale family)
loc_scale_model = TAVIE_loc_scale(fit_intercept=True, scale_X=False, scale_y=False, family="laplace", afunc=None, cfunc=None)

qr_model = TAVIE(fit_intercept=True, scale_X=False, scale_y=False)

## following is the Type II SSG example for 'binomial' (logistic regression), the other option is 'negbin' for negative-binomial regression
type_II_model = TAVIE_type_II(fint_intercept=True, scale_X=False, family="binomial")
```

**Note**: When initializing the TAVIE location-scale model for `laplace` or `student`, `afunc` and `cfunc` are computed in-built, whereas if a custom location-scale family is chosen, the corresponding callable functions for `afunc` and `cfunc` are to be provided to `TAVIE_loc_scale()`.

---

### Components of each TAVIE class

For each of the TAVIE class listed above, following are the components and their respective functionalities.


| Method Name                   | `TAVIE_loc_scale` | `TAVIE_type_II` | `TAVIE_QR` | Description                                                                 |
|-------------------------------|-------------------|-----------------|------------|-----------------------------------------------------------------------------|
| `fit()`                       | ✅                | ✅               | ✅         | Fits the TAVIE model to data using                                          |
| `get_TAVIE_means()`           | ✅                | ✅               | ✅         | Returns (and optionally displays) the TAVIE posterior means of parameters   |
| `get_variational_estimates()` | ✅                | ✅               | ✅         | Returns a dictionary of variational parameters (mean, covariance)           |
| `get_elbo()`                  | ✅                | ✅               | ✅         | Returns ELBO values tracked across iterations                               |

✅ = Supported. We give example usage of each of these above listed functions below for various SSG likelihoods considered in TAVIE.

---

### TAVIE in action for SSG Type I family: Laplace likelihood

We consider showing the utilities of each components in the `TAVIE_loc_scale()` class, particularly for the SSG Type I family having the *Laplace* likelihood of the form:

**Laplace error model**: $y_i = \beta_0 + \mathbf{X}_i^\top \boldsymbol{\beta} + \epsilon_i, \quad \text{where } \epsilon_i \sim \text{Laplace}(0, \sigma^2=\tau^{-2})$, for $i=1,2,\ldots,n$ with $f(\epsilon \mid \tau^2) = \frac{\sqrt{\tau^2}}{2} \exp\left( -\sqrt{\tau^2} \cdot |\epsilon| \right)$.

**Prior**: $(\boldsymbol{\beta}, \tau^2)$ is endowed upon with a prior as, $\boldsymbol{\beta}\mid \tau^2 \sim N(\boldsymbol{m}, \boldsymbol{V}/\tau^2)$ and $\tau^2\sim Gamma(a/2, b/2)$.

We first generate the data from the Laplace model with data parameters:
* $(n, p, \tau^2_{\text{true}}) = (10000, 5, 8)$,
* The design matrix $X\in \mathbb{R}^{p+1}$ comprise of entries from the *standard normal distribution* with the first column being $1_n$ automatically added by the `TAVIE_loc_scale()` class on choosing `fit_intercept=True`,
* $\beta_{\text{true}} = (\beta_0, \beta)\in \mathbb{R}^{p+1}$ is also generated from the *standard normal distribution*, and
* $\epsilon_i \sim Laplace(0, \tau_{\text{true}}^{-2})$.

```python
# Simulated data
n = 10000
p = 5
tau2 = 8

# Design matrix, true regression coefficients and response
X = np.random.normal(size=(n, p))
beta_true = np.random.normal(loc=0.0, scale=1.0, size=p+1)
error = np.random.laplace(size=n, loc=0.0, scale = 1/np.sqrt(tau2))
y = beta_true[0] + X @ beta_true[1:len(beta_true)] + error
```

Consequently, we initialize the TAVIE model and *fit* the initialized model using `fit()` for this particular Laplace likelihood:

```python
# Initialize the TAVIE model for laplace likelihood
laplace_model = TAVIE_loc_scale(family="laplace", fit_intercept=True) # choosing an intercept term
laplace_model.fit(X, y, verbose=True) # fit the TAVIE model
```

Now that the TAVIE model has been fit, we obtain the resultant *estimated TAVIE means* of $\beta_{\text{true}}$ and $\tau_{\text{true}}$ using the `get_TAVIE_means()` functionality along with printing them on the console using the `verbose=True` argument:

```python
laplace_model.get_TAVIE_means(verbose=True) # get the TAVIE estimates
```

If the user is interested to obtain the *variational estimates*, it can be done using `get_variational_estimates()`:

```python
# obtain the variational parameter estimates; use 'variational_est' as required
variational_est = laplace_model.get_variational_estimates()
```

To check the convergence diagnostics, we also have the `get_elbo()` functionality that could be used to obtain the *evidence lower bound* (ELBO) history over iterations:

```python
ELBO = laplace_model.get_elbo() # get the ELBO across iterations
```

<p align="center">
  <img src="assets/TAVIE_Laplace_ELBO_animation.gif" alt="TAVIE Laplace ELBO" width="600"/>
</p>

**Note**: TAVIE applied to other SSG likelihoods along with the utilities of each components in the different TAVIE classes have been illustrated in [TAVIE_examples.ipynb](./TAVIE_examples.ipynb).

---

### TAVIE vs other state-of-the-art variational inference algorithms

We present a bake-off of TAVIE against *mean-field variational inference* (MFVI) and *black-box variational inference* (BBVI) respectively for the *Student's-t* model. The MFVI algorithm is adopted from [Wand et al., 2011](https://projecteuclid.org/journals/bayesian-analysis/volume-6/issue-4/Mean-Field-Variational-Bayes-for-Elaborate-Distributions/10.1214/11-BA631.full) and is available in [CompetingMethods/mfvi.py](./CompetingMethods) along with the BBVI algorithm in [CompetingMethods/bbvi.py](./CompetingMethods).

First, we initialize the corresponding Student's-t TAVIE model:

```python
# TAVIE model initialization
t_model = TAVIE_loc_scale(family="student", fit_intercept=True)
```

Following the model initialization, we set the experimental data parameters and containers to store the $\ell_2$ errors between the true and estimated parameters ($\beta$ and $\tau^2$) from TAVIE, MFVI, and BBVI.

* The true regression coefficients are generated from $\beta \sim N(2, \tau^{-2})$, where $\tau^2 = 2$, and
* The degrees of freedom $\nu$ is set as $2$.

```python
# Experiment parameters
n = 10000
p = 5
nu_true = 2
tau2_true = 2
num_reps = 100

# Containers for metrics
mse_beta_MFVI = np.zeros(num_reps)
mse_beta_BBVI = np.zeros(num_reps)
mse_beta_TAVIE = np.zeros(num_reps)
mse_tau2_MFVI = np.zeros(num_reps)
mse_tau2_BBVI = np.zeros(num_reps)
mse_tau2_TAVIE = np.zeros(num_reps)
# for comparing the speed of the respective algorithms
time_TAVIE = np.zeros(num_reps)
time_MFVI = np.zeros(num_reps)
time_BBVI = np.zeros(num_reps)

# True beta vector
beta_true = np.random.normal(loc=2.0, scale=1.0, size=p+1)
```

We repeat this experiment for `num_reps=100` times and use the functionalities provided in the `TAVIE_loc_scale()` class to obtain the results:

```python
# Main loop with progress bar, repeating the experiment 'num_reps' times
for rep in trange(num_reps, desc="Repetitions"):
    # Generate synthetic data
    X = np.random.normal(size=(n, p))
    X_bbvi = np.column_stack((np.ones(n), X))
    
    error = t.rvs(df=nu_true, size=n) / np.sqrt(tau2_true)
    y = beta_true[0] + X @ beta_true[1:len(beta_true)] + error

    # TAVIE estimator
    t0 = perf_counter()
    t_model.fit(X, y, nu=nu_true, verbose=False) # fitting the TAVIE model for Student's-t
    time_TAVIE[rep] = perf_counter() - t0
    beta_est, tau2_est = t_model.get_TAVIE_means(verbose=False) # obtaining the TAVIE estimates
    mse_beta_TAVIE[rep] = np.mean((beta_est - beta_true)**2)
    mse_tau2_TAVIE[rep] = (tau2_est - tau2_true)**2
    

    # MFVI estimator
    t0 = perf_counter()
    beta_hat, sigma_sq_hat, nu_hat = MFVI_Student(X_bbvi, y, 
                                                  mu_beta=np.zeros(p+1), Sigma_beta=np.eye(p+1), 
                                                  A=2, B=2, nu_min=2.0, 
                                                  nu_max=20.0, tol=1e-6, verbose = False)
    time_MFVI = perf_counter() - t0
    beta_est2 = beta_hat
    tau2_est2 = 1/sigma_sq_hat
    mse_beta_MFVI[rep] = np.mean((beta_est2 - beta_true)**2)
    mse_tau2_MFVI[rep] = (tau2_est2 - tau2_true)**2

    # BBVI estimator
    t0 = perf_counter()
    res2 = BBVI_student(X_bbvi, y, nu=nu_true)
    time_BBVI[rep] = perf_counter() - t0
    beta_est3 = res2['beta_mean']
    tau2_est3 = res2['tau2_mean']
    mse_beta_BBVI[rep] = np.mean((beta_est3 - beta_true)**2)
    mse_tau2_BBVI[rep] = (tau2_est3 - tau2_true)**2
```

Finally, to show the performance of TAVIE, we present the boxplots of the $\ell_2$ errors between the true and estimated parameters ($\beta$ and $\tau^2$), along with the boxplots of the *run-times* across TAVIE, MFVI, and BBVI.

<p align="center">
  <img src="assets/TAVIE_MFVI_BBVI_MSE_Runtime_Comparison.gif" alt="TAVIE Laplace ELBO" width="600"/>
  <br><em>Comparison of TAVIE against MFVI and BBVI under the Student's-t model</em>
</p>

**Note**: For more TAVIE comparisons under different SSG likelihoods, please see [comparisons.ipynb](./comparisons.ipynb).

---

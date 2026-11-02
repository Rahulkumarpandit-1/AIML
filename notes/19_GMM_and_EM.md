# Gaussian Mixture Models (GMM) and Expectation-Maximization (EM)

## 1. What is GMM?

Gaussian Mixture Model (GMM) is a probabilistic, unsupervised learning model that represents data as a mixture of multiple Gaussian distributions.

It is used for:
- Clustering data.
- Modeling complex probability distributions.
- Estimating the probability that a sample belongs to each component.
- Detecting unusual observations when combined with appropriate likelihood thresholds.

Unlike K-Means, GMM can represent elliptical clusters and provide soft cluster assignments.

## 2. Gaussian Distribution

A Gaussian distribution, also called a normal distribution, is described by its mean and variance in one dimension.

- Mean (μ): the center of the distribution.
- Variance (σ²): the spread of the distribution.
- Standard deviation (σ): the square root of variance.

For multidimensional data, the covariance matrix (Σ) describes feature variances and relationships between features.

## 3. What is a Gaussian mixture?

A Gaussian mixture combines K Gaussian components.

The probability density is:

p(x) = Σ[k=1 to K] πₖ N(x | μₖ, Σₖ)

Where:
- K = number of Gaussian components.
- πₖ = mixture weight of component k.
- μₖ = mean of component k.
- Σₖ = covariance matrix of component k.
- N(x | μₖ, Σₖ) = Gaussian density for component k.

Mixture weights satisfy:

πₖ ≥ 0 for every component, and Σ[k=1 to K] πₖ = 1.

The mixture weights describe the relative contributions of the components to the probability distribution. They are not necessarily identical to the true proportions of real-world groups.

## 4. Hard Clustering vs Soft Clustering

### K-Means: hard clustering

Each point receives one cluster label.

For example: point A → cluster 0.

### GMM: soft clustering

Each point receives a probability for every component.

Example:

Component 0: 0.70  
Component 1: 0.25  
Component 2: 0.05

The probabilities sum to 1. The most likely component is component 0, but the model also assigns probability to the other components.

The probabilities express the model's uncertainty under its fitted distribution; they are not guaranteed to be perfectly calibrated.

## 5. Expectation-Maximization (EM)

EM is an iterative algorithm commonly used to estimate GMM parameters.

The main parameters are:
- Mixture weights.
- Component means.
- Component covariance matrices.

### E-step: Expectation

Estimate each sample's responsibility for every component using the current model parameters.

The responsibility is:

γᵢₖ = P(component k | sample xᵢ)

Using Bayes' rule:

γᵢₖ = [πₖ N(xᵢ | μₖ, Σₖ)] / [Σ[j=1 to K] πⱼ N(xᵢ | μⱼ, Σⱼ)]

For each sample i:

Σ[k=1 to K] γᵢₖ = 1.

### M-step: Maximization

Update the parameters using the responsibilities.

Effective component membership:

Nₖ = Σ[i=1 to N] γᵢₖ

Updated mixture weight:

πₖ = Nₖ / N

Updated mean:

μₖ = Σ[i=1 to N] γᵢₖ xᵢ / Nₖ

The covariance matrix is updated using the responsibility-weighted differences from the new mean:

Σₖ = Σ[i=1 to N] γᵢₖ (xᵢ − μₖ)(xᵢ − μₖ)ᵀ / Nₖ

The algorithm repeats E-step → M-step until convergence or an iteration limit is reached.

EM typically increases or maintains the data log-likelihood at each iteration under its standard assumptions, but it may converge to a local optimum rather than the global optimum.

## 6. Numerical intuition

Suppose there are two Gaussian components:

- Component A: mean 2.
- Component B: mean 8.
- Both have variance 1.
- Both have mixture weight 0.5.

For a point x = 3, the relative density terms are proportional to:

A: exp(−(3 − 2)² / 2) = exp(−0.5)

B: exp(−(3 − 8)² / 2) = exp(−12.5)

Normalize the terms:

P(A | x = 3) = exp(−0.5) / [exp(−0.5) + exp(−12.5)]

P(A | x = 3) ≈ 0.999994

The point is much more likely under component A.

In the E-step, these probabilities are calculated for all points. In the M-step, they help update the model parameters.

## 7. GMM vs K-Means

| Property | K-Means | GMM |
|---|---|---|
| Assignment | Hard | Soft probabilities available |
| Objective | Minimize within-cluster squared distances | Maximize data likelihood |
| Cluster representation | Centroids | Gaussian distributions |
| Cluster shape | Typically favors spherical groups | Can represent elliptical groups |
| Uncertainty | No component probabilities | Component probabilities |
| Training | Centroid updates | Usually EM |

Neither algorithm is universally better. The suitable method depends on the data distribution and task.

## 8. Scikit-learn implementation

File: `machine_learning/06_unsupervised_learning/06_gmm_clustering.py`

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

X, _ = make_blobs(
    n_samples=400,
    centers=3,
    cluster_std=[1.0, 1.5, 0.8],
    random_state=42
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gmm = GaussianMixture(
    n_components=3,
    covariance_type="full",
    random_state=42,
    n_init=5
)

gmm.fit(X_scaled)

labels = gmm.predict(X_scaled)
probabilities = gmm.predict_proba(X_scaled)

print("Labels:", labels[:10])
print("First point probabilities:", probabilities[0])
print("Probability sum:", probabilities[0].sum())
print("Mixture weights:", gmm.weights_)
print("Converged:", gmm.converged_)
print("Iterations:", gmm.n_iter_)
print("Average log-likelihood:", gmm.score(X_scaled))

plt.figure(figsize=(8, 6))
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap="viridis",
    alpha=0.7
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Gaussian Mixture Model Clustering")
plt.show()
```

## 9. Important attributes and methods

- `fit(X)`: estimate the mixture parameters.
- `predict(X)`: return the most likely component label.
- `predict_proba(X)`: return component probabilities for every sample.
- `weights_`: learned mixture weights.
- `means_`: learned component means.
- `covariances_`: learned covariance matrices.
- `converged_`: whether the algorithm reports convergence.
- `n_iter_`: number of iterations performed.
- `score(X)`: average log-likelihood per sample.

Cluster labels are arbitrary identifiers. Label 0 is not inherently better or more important than label 1.

## 10. Covariance types

The `covariance_type` parameter controls the allowed covariance structure.

- `full`: each component has its own unrestricted covariance matrix.
- `tied`: all components share one covariance matrix.
- `diag`: each component has a diagonal covariance matrix.
- `spherical`: each component has one variance value shared across dimensions.

More flexible covariance models can represent richer shapes but may require more data and computation.

## 11. Selecting the number of components

The number of components is set through `n_components`.

Possible approaches include:
- Comparing model performance and interpretability.
- Using the Akaike Information Criterion (AIC).
- Using the Bayesian Information Criterion (BIC).

For both AIC and BIC, smaller values are preferred when comparing candidate models fitted to the same dataset under comparable assumptions.

BIC penalizes model complexity more strongly than AIC in many common settings.

Do not select a model using training likelihood alone, because adding components can increase training likelihood even when the additional complexity is not useful.

## 12. Advantages

- Provides probabilities rather than only hard labels.
- Models elliptical distributions.
- Can represent overlapping groups.
- Offers a probabilistic interpretation of clustering.

## 13. Limitations

- Requires choosing the number of components or comparing candidate models.
- EM can converge to a local optimum.
- Results depend on initialization and the data.
- Flexible covariance matrices may overfit small datasets.
- A Gaussian mixture may be unsuitable for highly irregular cluster shapes.
- Soft probabilities depend on the model assumptions and are not guaranteed to be calibrated.

## 14. Practical experiments

1. Change `n_components` from 3 to 2 and compare the visualization.
2. Inspect `predict_proba()` for several samples.
3. Find samples whose maximum component probability is low.
4. Check `converged_` and `n_iter_`.
5. Compare candidate component counts using AIC and BIC.
6. Try `covariance_type="full"` and `"diag"` and compare results.

## 15. Key takeaways

- GMM models data as a mixture of Gaussian components.
- K-Means makes hard assignments; GMM provides component probabilities.
- EM alternates between estimating responsibilities and updating parameters.
- Component probabilities for each sample sum to 1.
- Use likelihood, AIC, BIC, and practical evaluation to compare candidate models.

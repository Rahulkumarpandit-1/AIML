# SVD and Its Applications

## 1. Definition

Singular Value Decomposition (SVD) decomposes a matrix into three matrices:

**Formula:** A = U × Σ × Vᵀ

- **U:** left singular vectors.
- **Σ:** diagonal matrix of singular values.
- **Vᵀ:** transpose of the right singular-vector matrix.
- **Singular values:** non-negative values that indicate the strength of each component.

## 2. Low-Rank Approximation

Low-rank approximation reconstructs a matrix using only its largest singular values.

- More components generally preserve more information.
- Fewer components reduce representation size but lose information.
- Reconstruction error measures the difference between the original and approximated matrices.

## 3. Applications

1. **Image compression:** approximate image matrices with fewer components.
2. **NLP / Latent Semantic Analysis:** reduce document-term matrices to uncover useful latent patterns.
3. **Recommendation systems:** approximate user-item matrices using latent factors.
4. **Dimensionality reduction:** represent data with fewer dimensions.
5. **Noise reduction:** small components may sometimes represent noise, but removing them is not always beneficial.

## 4. SVD and PCA

PCA is a dimensionality-reduction method. SVD is a matrix factorization method that can be used to compute PCA.

For centered data X:

- SVD: X = U × Σ × Vᵀ
- Principal directions are given by the right singular vectors.
- Explained variance for component i = σᵢ² / (n − 1), using sample covariance.
- PCA typically centers features before computing components.
- SVD can also be applied to uncentered matrices.

## 5. Python

```python
U, s, Vt = np.linalg.svd(A, full_matrices=False)
```

For k components:

```python
A_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```

Reconstruction error:

```python
error = np.linalg.norm(A - A_approx)
```

## 6. Limitations

- Reducing dimensions can discard useful information.
- Full SVD may be computationally expensive for large matrices.
- The components are mathematical patterns and do not automatically have a human-interpretable meaning.
- Centering, scaling, and the choice of component count depend on the task.
- Truncated SVD is particularly useful for large sparse matrices, such as text-feature matrices.

## 7. Key Takeaway

SVD decomposes a matrix into components ordered by singular value. Keeping the strongest components creates a lower-rank approximation that can be useful for compression, dimensionality reduction, text analysis, and recommendation systems.

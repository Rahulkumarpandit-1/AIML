
import numpy as np

# Each row can represent a document; each column represents a feature.
A = np.array([
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 1]
], dtype=float)

# SVD decomposition
U, singular_values, Vt = np.linalg.svd(A, full_matrices=False)

print("Original matrix:")
print(A)

print("\nSingular values:")
print(singular_values)

# Reconstruct using only the largest k singular values
for k in [1, 2, 4]:
    A_approx = U[:, :k] @ np.diag(singular_values[:k]) @ Vt[:k, :]

    error = np.linalg.norm(A - A_approx)

    print(f"\nUsing {k} component(s):")
    print(np.round(A_approx, 2))
    print("Reconstruction error:", round(error, 4))

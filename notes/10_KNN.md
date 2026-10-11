# K-Nearest Neighbors (KNN)

## 1. What is KNN?

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for classification and regression.

- **Classification:** predicts the most common class among the K nearest training samples.
- **Regression:** commonly predicts the average target value among the K nearest training samples.

KNN is distance-based: it uses the stored training examples to make predictions.

## 2. How KNN classification works

1. Store the training features and labels.
2. Calculate the distance between a new sample and the training samples.
3. Select the K nearest samples.
4. Let their labels vote.
5. Predict the majority class.

**Important:** A new sample is compared against stored training data, not against test samples.

### Euclidean distance

For two points with features x and z:

d(x, z) = √[(x₁ − z₁)² + (x₂ − z₂)² + ... + (xₙ − zₙ)²]

Plain text: Distance equals the square root of the sum of squared differences between corresponding features.

## 3. How K affects the decision boundary

- **Small K, such as K = 1:** predictions depend on very few neighbors. Boundaries can be irregular and sensitive to noise. Overfitting is more likely.
- **Moderate K, such as K = 3 or K = 5:** more neighbors contribute to each prediction, often producing a more balanced boundary.
- **Larger K, such as K = 15:** the boundary is generally smoother because predictions consider more neighbors. If K is too large, the model may underfit.

Increasing K generally smooths the boundary, but it does not guarantee better accuracy.

Choose K using validation data or cross-validation, not by repeatedly checking test accuracy.

## 4. Why do we use StandardScaler?

KNN relies on distances. If one feature ranges from 0 to 10,000 while another ranges from 0 to 1, the larger-range feature can dominate distance calculations.

`StandardScaler` standardizes features using the mean and standard deviation calculated during fitting.

A pipeline ensures that the scaler is fitted on training data and the same transformation is applied to test data and future inputs.

## 5. Scikit-learn pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

model = Pipeline([
    ("standardscaler", StandardScaler()),
    ("kneighborsclassifier", KNeighborsClassifier(n_neighbors=3))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = model.score(X_test, y_test)

print("Test accuracy:", accuracy)
```

### Understanding the code

- `Pipeline`: combines preprocessing and model steps.
- `StandardScaler()`: scales the features.
- `KNeighborsClassifier(n_neighbors=3)`: classifies using three nearest neighbors.
- `fit(X_train, y_train)`: fits the scaler on training features and fits the KNN model.
- `predict(X_test)`: predicts labels for test samples.
- `score(X_test, y_test)`: returns classification accuracy on the supplied test set.

The names `standardscaler` and `kneighborsclassifier` are chosen names for the pipeline steps. They are not mandatory names.

## 6. Visualizing decision boundaries

A **decision boundary** separates regions where a classifier predicts different classes.

For a two-dimensional dataset:

1. Create a grid of points covering the plot.
2. Predict the class at each grid point.
3. Color each region according to the predicted class.
4. Plot the training samples on top.
5. Repeat for different K values.

In our visualization exercise:

- `make_moons()` creates a two-class dataset.
- `np.meshgrid()` creates grid coordinates.
- `model.predict(grid)` predicts classes for grid points.
- `contourf()` colors the decision regions.
- `model.score(X_test, y_test)` displays test accuracy.

The colored regions show predictions across the feature space; they are not extra training samples.

## 7. Training, validation, and test data

- **Training set:** used to fit the model.
- **Validation data or cross-validation:** used to compare choices such as K.
- **Test set:** reserved for final evaluation after model selection.

Do not repeatedly select K based on test accuracy. Use validation data or cross-validation to choose hyperparameters.

## 8. Strengths and limitations

### Strengths
- Simple to understand and implement.
- Can learn non-linear decision boundaries.
- Requires little explicit model-fitting computation.

### Limitations
- Prediction can be slow on large datasets.
- Sensitive to feature scaling and irrelevant features.
- Can struggle in high-dimensional spaces.
- Performance depends on K and the distance metric.
- Imbalanced classes can affect majority voting.

## 9. Quick revision

- KNN predicts using nearby training samples.
- K is the number of neighbors that vote in classification.
- Scaling matters because KNN uses distances.
- A pipeline keeps preprocessing and prediction steps together.
- Small K often produces a more complex boundary.
- Larger K generally produces a smoother boundary.
- Use validation or cross-validation to choose K.
- Reserve the test set for final evaluation.

1. Numerical dry run
Suppose the new student is represented by features (4, 3), and three nearest training students have these labels:
Neighbor	Distance	Label
A	1.0	Pass
B	1.4	Fail
C	2.0	Pass
With \(K=3\), two neighbors vote Pass and one votes Fail, so the prediction is Pass.
2. Important KNN limitation
KNN can be sensitive to imbalanced classes. If one class is much more common, majority voting may favor that class. Distance-weighted voting, class-specific evaluation metrics, or other approaches may help, depending on the problem.
3. What our visualization taught us
The four plots compared \(K=1,3,5,15\) on a two-dimensional dataset. Smaller K generally produces more intricate boundaries, while larger K generally smooths them. The displayed test accuracy lets us compare performance, but we should use validation data or cross-validation to select K.

##  Practice questions

1. Why can an unscaled feature dominate KNN distance?
2. What generally happens to the decision boundary as K increases from 1 to 15?
3. Why should the scaler be fitted only on training data?
4. What does `model.score(X_test, y_test)` return for `KNeighborsClassifier`?
5. Why should K be selected using validation data or cross-validation rather than repeatedly checking test accuracy?
# Data Preprocessing

## 1. What is Data Preprocessing?

Data preprocessing is the process of converting raw data into a clean and suitable format for machine learning algorithms.

Typical steps:

```text
Raw Data
   ↓
Cleaning
   ↓
Encoding
   ↓
Scaling
   ↓
Processed Data
   ↓
ML Model
```

---

## 2. Numerical and Categorical Features

### Numerical Feature

A feature containing quantitative numerical values.

Examples:

```text
age
salary
experience
height
```

### Categorical Feature

A feature containing categories or labels.

Examples:

```text
city → Delhi, Mumbai, Guwahati
department → CSE, ECE, ME
```

Categorical features usually need to be encoded before being given to ML models.

---

# 3. Encoding

Encoding converts categorical values into numerical representations that ML algorithms can process.

## 3.1 One-Hot Encoding

Used when categories **do not have a meaningful order**.

Example:

```text
city
Delhi
Mumbai
Guwahati
```

becomes:

```text
city_Delhi  city_Guwahati  city_Mumbai
     1            0              0
     0            0              1
     0            1              0
```

### Why?

Do not encode:

```text
Delhi = 1
Mumbai = 2
Guwahati = 3
```

because this creates an artificial numerical order.

### Scikit-learn

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False)

encoded = encoder.fit_transform(df[["city"]])
```

Useful method:

```python
encoder.get_feature_names_out(["city"])
```

---

## 3.2 Ordinal Encoding

Used when categories have a **meaningful order**.

Example:

```text
Low < Medium < High
```

can become:

```text
Low    → 0
Medium → 1
High   → 2
```

### Scikit-learn

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(
    categories=[["Low", "Medium", "High"]]
)

encoded = encoder.fit_transform(df[["level"]])
```

### Important rule

Use:

- **One-Hot Encoding** → unordered categories
- **Ordinal Encoding** → ordered categories

---

# 4. Feature Scaling

Feature scaling changes the numerical range or distribution of features so that features with very different scales are comparable.

Example:

```text
age    → 20–40
salary → 20,000–100,000
```

Some ML algorithms are sensitive to differences in feature scale.

Scaling is particularly important for algorithms based on distance, magnitude, or optimization.

Examples:

- KNN
- SVM
- Logistic Regression
- Neural Networks

Tree-based models such as Decision Trees and Random Forests generally do not require feature scaling.

---

# 5. Standardization

Standardization transforms a feature so that it has approximately:

```text
mean = 0
standard deviation = 1
```

### Formula

```text
z = (x - μ) / σ
```

Where:

- `x` = original value
- `μ` = mean of the feature
- `σ` = standard deviation
- `z` = standardized value

Example:

```text
20 → -1.41
25 → -0.71
30 →  0.00
35 →  0.71
40 →  1.41
```

Interpretation:

`-1.41` means the value is approximately 1.41 standard deviations below the mean.

### Scikit-learn

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

---

# 6. Min-Max Scaling

Min-Max Scaling transforms values to a specified range, commonly `[0, 1]`.

### Formula

```text
x' = (x - xmin) / (xmax - xmin)
```

Example:

```text
20 → 0.00
25 → 0.25
30 → 0.50
35 → 0.75
40 → 1.00
```

### Scikit-learn

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
```

### StandardScaler vs MinMaxScaler

| StandardScaler | MinMaxScaler |
|---|---|
| Centers data around 0 | Usually maps data to 0–1 |
| Uses mean and standard deviation | Uses minimum and maximum |
| Less directly affected by min/max values | Sensitive to extreme values |

Do not choose a scaler blindly; the appropriate choice depends on the algorithm and data.

---

# 7. fit(), transform(), fit_transform()

These methods are fundamental to preprocessing.

### `fit()`

Learns required information from the data.

Examples:

```text
StandardScaler → learns mean and standard deviation
OneHotEncoder → learns categories
Imputer → learns replacement statistics
```

### `transform()`

Uses the information already learned to transform data.

### `fit_transform()`

Performs both operations:

```text
fit()
  ↓
learn
  ↓
transform()
```

Example:

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

# 8. ColumnTransformer

`ColumnTransformer` applies different preprocessing operations to different columns.

Example:

```text
age, salary → StandardScaler
city        → OneHotEncoder
```

### Code

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["age", "salary"]),
    ("cat", OneHotEncoder(), ["city"])
])
```

Then:

```python
X_processed = preprocessor.fit_transform(X)
```

### Purpose

It allows multiple preprocessing transformations to be organized in one preprocessing object.

---

# 9. Pipeline

A `Pipeline` chains preprocessing steps and a model into one workflow.

### Code

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LogisticRegression())
])
```

Training:

```python
model_pipeline.fit(X_train, y_train)
```

Prediction:

```python
predictions = model_pipeline.predict(X_test)
```

### Workflow

```text
X_train
   ↓
Preprocessing
   ↓
Processed X_train
   ↓
Model
   ↓
Prediction
```

The Pipeline helps keep preprocessing and model training consistent and reduces the chance of applying transformations incorrectly.

---

# 10. Data Leakage

**Data leakage** occurs when information that should be unavailable during model training influences the training process.

This can cause unrealistically good evaluation results and poor performance on genuinely unseen data.

### Correct rule

Anything that learns information from the data should learn it from the **training set only**.

Examples:

```text
Scaler
Imputer
Encoder
Feature Selection
Model
```

### Correct workflow

```text
Full Dataset
     ↓
Train/Test Split
     ↓
X_train ──→ fit preprocessing
              ↓
          learned parameters
              ↓
X_train ──→ transform
X_test  ──→ transform
              ↓
          Model prediction
```

### Wrong

```text
Full Dataset
     ↓
Fit preprocessing
     ↓
Train/Test Split
```

The preprocessing has already seen information from the test set.

---

# 11. Target Leakage

Target leakage occurs when a feature contains information about the target that would not legitimately be available at prediction time.

Example:

Trying to predict whether a student will pass **before the final exam**, while using `final_exam_score` as a feature.

```text
Features:
- hours_studied
- attendance
- final_exam_score  ← leakage

Target:
- passed
```

The final exam score is known after the exam, so it would not be available when making the intended prediction.

### Rule

When selecting features, ask:

> Would this information genuinely be available at prediction time?

If not, it may cause target leakage.

---

# 12. Train/Test Preprocessing Rule

The test set must remain unseen during learning.

Correct:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model_pipeline.fit(X_train, y_train)

predictions = model_pipeline.predict(X_test)
```

The Pipeline fits preprocessing using training data and then applies the learned transformations to the test data.

---

# 13. Practical Rules to Remember

### Encoding

```text
Unordered categories → One-Hot Encoding
Ordered categories   → Ordinal Encoding
```

### Scaling

```text
StandardScaler → mean ≈ 0, standard deviation ≈ 1
MinMaxScaler   → usually range 0–1
```

### Learning vs applying

```text
fit       → learn
transform → apply
fit_transform → learn + apply
```

### Leakage prevention

```text
Split first
↓
Fit preprocessing on training data
↓
Transform training and test data
↓
Train model
↓
Evaluate on test data
```

### Pipeline

```text
Preprocessing + Model
        ↓
     Pipeline
```

---

# 14. Required Libraries

```python
import pandas as pd

from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
```

---

# 15. Common Mistakes

- Using numerical labels for unordered categories instead of One-Hot Encoding.
- Using Ordinal Encoding when categories have no meaningful order.
- Scaling categorical features unnecessarily.
- Fitting a scaler on the complete dataset before splitting.
- Fitting an encoder on the test set.
- Using information that would not be available at prediction time.
- Assuming every ML algorithm requires scaling.
- Evaluating a model on the same data used for training.
- Treating a tiny practice dataset as evidence that a model is genuinely accurate.

---

## Core Mental Model

```text
Raw Dataset
     ↓
Train/Test Split
     ↓
Identify feature types
     ↓
Numerical ──────→ Scaling
Categorical ────→ Encoding
     ↓
ColumnTransformer
     ↓
Pipeline
     ↓
ML Model
     ↓
Prediction
     ↓
Evaluation
```
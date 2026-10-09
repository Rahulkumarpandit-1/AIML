# Feature Engineering

## 1. What is Feature Engineering?

A **feature** is an input variable used by a machine learning model to make predictions.

**Feature engineering** is the process of creating, transforming, or combining raw data into useful features that may help a machine learning model learn better patterns.

### Core idea

```text
Raw data
   ↓
Feature Engineering
   ↓
Useful features
   ↓
Preprocessing
   ↓
ML Model
   ↓
Prediction
```

The goal is **not to create more columns**. The goal is to create better representations of useful information.

---

## 2. Feature vs Target

### Feature
An input variable used to make a prediction.

Examples:

- hours studied
- attendance
- salary
- experience

### Target
The value the model is trying to predict.

Example:

```text
Features → hours, attendance
Target   → pass/fail
```

A target should not accidentally become an input feature through leakage.

---

## 3. Common Feature Engineering Techniques

### A. Combination

Combine existing features to create a new feature.

Example:

```python
df["total_activity"] = df["hours"] + df["attendance"]
```

Use this when the combination has meaningful interpretation.

---

### B. Ratio

Divide one feature by another to represent a relationship.

Example:

```python
df["salary_per_experience"] = df["salary"] / df["experience"]
```

Example:

```text
salary = 60000
experience = 5

salary_per_experience = 12000
```

Always check for division by zero.

---

### C. Transformation

Transform a feature into another representation.

Example:

```python
import numpy as np

df["log_salary"] = np.log1p(df["salary"])
```

`np.log1p(x)` calculates:

```text
log(1 + x)
```

Log transformation can reduce the effect of very large values and is often useful for highly skewed numerical features.

Do not automatically apply log transformation to every feature.

---

### D. Interaction Feature

An interaction feature represents the combined effect of multiple features.

Example:

```python
df["study_attendance"] = df["hours"] * df["attendance"]
```

If:

```text
hours = 8
attendance = 90
```

then:

```text
study_attendance = 720
```

Interaction is different from correlation.

- **Correlation:** measures how variables move together.
- **Interaction:** represents whether the effect of one feature may depend on another feature.

---

### E. Date/Time Feature Extraction

Raw dates can be converted into useful components.

First convert strings to datetime:

```python
df["date"] = pd.to_datetime(df["date"])
```

Then extract components:

```python
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
```

`dayofweek` uses:

```text
Monday    → 0
Tuesday   → 1
Wednesday → 2
Thursday  → 3
Friday    → 4
Saturday  → 5
Sunday    → 6
```

Weekend feature:

```python
df["is_weekend"] = df["date"].dt.dayofweek >= 5
```

For timestamps, time can also be extracted:

```python
df["hour"] = df["datetime"].dt.hour
```

Only extract date/time components that have a reasonable connection to the prediction task.

---

# 4. Feature Engineering vs Preprocessing vs Feature Selection

These three concepts are related but have different purposes.

### Feature Engineering

**Create or transform features.**

```text
hours + attendance → study_score
```

Think:

**CREATE / TRANSFORM**

### Feature Selection

**Choose which existing features should be used.**

```text
hours
attendance
study_score
```

while removing an irrelevant feature such as:

```text
random_id
```

Think:

**KEEP / REMOVE**

### Data Preprocessing

**Prepare features for the ML algorithm.**

Examples:

```text
city → One-Hot Encoding
salary → StandardScaler
```

Think:

**PREPARE / CONVERT**

### Simple mental model

```text
Feature Engineering → CREATE
Feature Selection   → CHOOSE
Preprocessing       → PREPARE
```

---

# 5. Feature Engineering Is an Experiment

Creating a feature does not guarantee that the model will improve.

Correct mindset:

```text
Create feature
      ↓
Train/evaluate model
      ↓
Compare performance
      ↓
Keep if useful
```

A feature can sometimes:

- improve performance
- have no meaningful effect
- make the model worse
- introduce leakage

Therefore, engineered features should be evaluated rather than blindly kept.

---

# 6. Target Leakage

**Target leakage** occurs when a feature contains information about the target that would not legitimately be available when the prediction is supposed to be made.

Example:

Suppose we want to predict whether a student will pass **before the final exam**.

Using:

```text
final_exam_score
```

as an input feature would be leakage because the final score is not available at prediction time.

### Important question

> Would this information genuinely be available at prediction time?

If not, it should not be used as a feature.

---

# 7. Common Mistakes

### Creating features without a reason

More columns do not automatically mean a better model.

### Ignoring division by zero

For example:

```python
df["salary_per_experience"] = df["salary"] / df["experience"]
```

can cause problems when `experience = 0`.

### Using target information

Never create features using information that would only become available after the target is known.

### Confusing scaling with feature engineering

```python
df["log_salary"] = np.log1p(df["salary"])
```

is feature engineering.

```python
StandardScaler().fit_transform(X)
```

is preprocessing.

### Poor feature names

Prefer descriptive names such as:

```text
salary_per_experience
study_attendance
is_weekend
```

rather than vague names such as:

```text
feature1
efficiency
new_column
```

unless those names have a clearly defined meaning.

---

# 8. Important Libraries

```python
import pandas as pd
import numpy as np
```

Scikit-learn is also commonly used later for preprocessing and feature-selection techniques:

```python
from sklearn.preprocessing import StandardScaler
```

---

# 9. Core Mental Model

Remember:

```text
RAW DATA
   ↓
Feature Engineering
   ↓
Create useful representations
   ↓
Feature Selection
   ↓
Choose useful features
   ↓
Preprocessing
   ↓
Encode / scale / transform
   ↓
Model Training
   ↓
Evaluation
```

### Most important rule

> Create features because they represent useful information, not simply because you can create them.
# 01 — Machine Learning Fundamentals

## 1. What Is Machine Learning?

Machine Learning (ML) is a branch of Artificial Intelligence (AI) that enables computers to learn patterns from data and use those patterns to make predictions or decisions.

In traditional programming, we write explicit rules to produce an output. In Machine Learning, we provide examples and allow an algorithm to learn patterns from them.

**Example: House Price Prediction**

Suppose we have this dataset:

| Area (sq. ft.) | Bedrooms | Price (₹) |
|---:|---:|---:|
| 800 | 2 | 25,00,000 |
| 1,000 | 2 | 32,00,000 |
| 1,200 | 3 | 40,00,000 |
| 1,500 | 3 | 50,00,000 |

An ML algorithm can learn a relationship between house characteristics and prices. We can then use the trained model to predict the price of a new house.

The prediction is not guaranteed to be correct. Its quality depends on the data, the model, and the evaluation process.

---

## 2. AI vs ML vs Deep Learning

### Artificial Intelligence (AI)

The broad field of creating systems that perform tasks associated with intelligence, such as reasoning, planning, perception, and language understanding.

### Machine Learning (ML)

A subset of AI in which algorithms learn patterns from data.

### Deep Learning (DL)

A subset of ML that uses neural networks containing multiple layers.

**Relationship:**

AI → Machine Learning → Deep Learning

Generative AI systems, such as large language models, commonly use deep learning.

---

## 3. Important Machine Learning Terminology

### Dataset

A collection of examples used to train, validate, or test an ML system.

### Sample / Observation

One individual example in a dataset.

For example, one row describing a house is one sample.

### Feature

An input variable used by a model to make a prediction.

Examples:
- House area
- Number of bedrooms
- Age of a house

Features are commonly represented by \(X\).

### Target / Label

The value the model is supposed to predict.

For house price prediction, the target is the price.

The target is commonly represented by \(y\).

### Model

A mathematical function that maps input features to predictions.

\[
\hat{y}=f(X)
\]

Here:
- \(X\): input features
- \(f\): learned model
- \(\hat{y}\): predicted value

### Algorithm

A method used to learn patterns or solve a problem.

Examples include Linear Regression, Decision Trees, and K-Nearest Neighbors.

**Difference:** The algorithm is the learning procedure; the model is what results after training.

### Parameters

Values learned by the model during training.

For Linear Regression, these include the weight and bias.

### Hyperparameters

Settings chosen before or during the training process that control how the model learns.

Examples:
- Learning rate
- Maximum tree depth
- Number of neighbors in KNN

Hyperparameters are not the same as learned model parameters.

### Prediction

The output produced by a model for a given input.

### Inference

Using a trained model to make predictions on new inputs.

---

## 4. Types of Machine Learning

There are three major learning categories to understand first.

### 4.1 Supervised Learning

The model learns from labeled examples, meaning the training data includes both inputs and the correct target values.

**Example:** Predicting house prices using historical house data.

Two major supervised learning tasks are:

**A. Regression**

Regression predicts a numerical quantity.

Examples:
- House prices
- Temperature
- Sales revenue
- Delivery time

Common algorithms:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

**B. Classification**

Classification predicts a category or class.

Examples:
- Spam or not spam
- Fraudulent or legitimate transaction
- Cat or dog
- Customer will leave or stay

Common algorithms:
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine

Important: Despite its name, Logistic Regression is generally used for classification.

### 4.2 Unsupervised Learning

The model works with data that does not have target labels and attempts to discover structure or patterns.

**Example:** Grouping customers according to purchasing behavior.

Common tasks:

**Clustering:** Grouping similar data points.

Algorithms:
- K-Means
- DBSCAN
- Hierarchical Clustering

**Dimensionality Reduction:** Representing data with fewer variables while retaining useful information.

Algorithms:
- Principal Component Analysis (PCA)

Unsupervised learning does not automatically tell us what a discovered group means. We must interpret the results.

### 4.3 Reinforcement Learning

An agent learns to choose actions by interacting with an environment and receiving rewards or penalties.

Important terms:
- **Agent:** The learner or decision-maker.
- **Environment:** The world in which the agent operates.
- **Action:** A decision made by the agent.
- **Reward:** Feedback associated with an action.
- **Policy:** The strategy used to choose actions.

Examples:
- Game-playing agents
- Robot control
- Resource allocation

Reinforcement Learning is different from supervised learning because the feedback is generally based on rewards rather than a supplied correct label for every action.

---

## 5. How a Machine Learning System Works

A typical supervised ML workflow:

1. Define the problem.
2. Collect relevant data.
3. Inspect and understand the data.
4. Clean the data and preprocess its features.
5. Split the data into training, validation, and test sets.
6. Select a suitable algorithm.
7. Train the model.
8. Evaluate the model.
9. Tune the model if needed.
10. Evaluate the final approach on the test set.
11. Deploy the model if it meets the requirements.
12. Monitor performance after deployment.

**Example: Spam Detection**

- Input: Email text and possibly other email features.
- Target: Spam or not spam.
- Training: Learn patterns from labeled emails.
- Evaluation: Check how well the model classifies unseen emails.
- Inference: Classify a newly received email.

The model should be evaluated on data that was not used to fit its parameters.

---

## 6. Training, Validation, and Test Data

We commonly divide a dataset into separate subsets.

### Training Set

Used to fit the model and learn its parameters.

### Validation Set

Used to compare models, tune hyperparameters, and make development decisions.

### Test Set

Used for a final, relatively unbiased evaluation after model choices are complete.

**Example split:**

- 70% training
- 15% validation
- 15% testing

These percentages are examples, not mandatory rules. The appropriate split depends on the dataset size, task, and data structure.

### Why not evaluate only on training data?

A model can perform very well on training examples but poorly on new examples. Evaluating only on training data can give an overly optimistic picture of performance.

### Important Rule: Avoid Data Leakage

Data leakage occurs when information that would not legitimately be available at training or prediction time influences the model or its evaluation.

Examples:
- Scaling the entire dataset before splitting it.
- Using future information to predict the past.
- Including a feature that directly reveals the target.

**Correct approach:** Split the data first. Fit preprocessing steps on the training data only, then apply the fitted transformations to validation, test, and future data.

For time-series problems, use time-aware splits instead of randomly mixing past and future observations.

---

## 7. Underfitting and Overfitting

### Underfitting

A model is too simple to learn the important patterns in the data.

Typical signs:
- High training error
- High validation error

Possible solutions:
- Use a more expressive model.
- Add meaningful features.
- Reduce excessive regularization.
- Train appropriately.

### Overfitting

A model learns the training data too closely, including noise or accidental patterns, and does not generalize well to new data.

Typical signs:
- Very low training error
- Much higher validation error

Possible solutions:
- Collect more representative data.
- Simplify the model.
- Apply regularization.
- Use feature selection when appropriate.
- Use cross-validation for model selection.
- Stop training at an appropriate point for iterative models.

### Good Generalization

A model generalizes when it performs well on genuinely unseen data from the intended prediction environment.

The goal is not simply to memorize training examples. The goal is to learn patterns that remain useful on new examples.

---

## 8. Bias and Variance

Bias and variance describe different sources of prediction error.

### High Bias

The model makes overly simplistic assumptions and misses important patterns.

High bias is often associated with underfitting.

### High Variance

The model is too sensitive to the particular training dataset and may learn noise.

High variance is often associated with overfitting.

A useful objective is to balance model complexity and generalization.

Increasing model complexity can reduce bias but may increase variance. The relationship depends on the dataset and learning algorithm.

---

## 9. Features, Labels, and Feature Engineering

Suppose we want to predict whether a student will pass an exam.

Possible features:
- Study hours
- Attendance
- Previous marks

Target:
- Pass or fail

Feature engineering means creating, transforming, or selecting useful input variables based on domain knowledge.

Examples:
- Extracting the month from a date.
- Calculating a customer's average monthly spending.
- Converting text into numerical representations.
- Creating an age group from age, when appropriate.

Useful features can improve performance, but poorly designed features can introduce leakage or unnecessary complexity.

---

## 10. Data Preprocessing

Real-world data often contains missing values, categorical values, different numerical scales, and other problems.

Preprocessing transforms data into a form suitable for an ML algorithm.

### Missing Values

Some rows may have missing age, salary, or other fields.

Possible approaches:
- Remove rows when justified.
- Fill numerical values with the median or mean.
- Fill categorical values with the most frequent category.
- Use more advanced imputation methods when needed.

The correct method depends on why values are missing and how much information is lost.

### Categorical Encoding

Many ML algorithms require numerical inputs.

For example:

| City | Encoded representation using one-hot encoding |
|---|---|
| Delhi | [1, 0, 0] |
| Mumbai | [0, 1, 0] |
| Guwahati | [0, 0, 1] |

One-hot encoding creates separate binary features for categories.

Do not assign arbitrary integer values to nominal categories when those integers could falsely imply an ordering.

### Feature Scaling

Features may have different numerical ranges.

Example:
- Age: 18–60
- Salary: 20,000–200,000

Scaling can help algorithms such as KNN, SVM, and gradient-based models.

Common methods:
- Standardization
- Min-max scaling

Tree-based models generally do not require feature scaling in the same way distance-based or gradient-based algorithms often do.

### Preprocessor

A preprocessor is a step or collection of steps that transforms raw features before they enter the model.

Examples:
- Fill missing values.
- Scale numerical columns.
- Encode categorical columns.

### Pipeline

A pipeline chains preprocessing and model training into a single workflow.

A typical scikit-learn pipeline might be:

1. Impute missing values.
2. Scale numerical features or encode categorical features.
3. Train a model.

A pipeline helps keep training and inference consistent and reduces the risk of preprocessing leakage when used correctly.

---

## 11. Loss Functions and Evaluation Metrics

These concepts are related but not identical.

### Loss Function

A numerical function that measures the error associated with a model's predictions during training.

A learning algorithm can use this value to determine how to adjust the model.

For regression, one common loss is Mean Squared Error (MSE):

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
\]

Here:
- \(n\): number of observations
- \(y_i\): actual value
- \(\hat y_i\): predicted value

Squaring makes large errors contribute more strongly.

### Evaluation Metric

A measure used to assess how well the model performs for the task.

Common regression metrics:
- MAE
- MSE
- RMSE
- \(R^2\)

Common classification metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The right metric depends on the problem. Accuracy alone can be misleading when classes are highly imbalanced.

---

## 12. Gradient Descent

Gradient Descent is an optimization algorithm used to minimize a loss function by adjusting model parameters.

For a parameter \(w\), the update rule is:

\[
w_{\text{new}}=w_{\text{old}}-\alpha\frac{dL}{dw}
\]

Where:
- \(w\): model parameter
- \(L\): loss function
- \(\frac{dL}{dw}\): gradient of the loss with respect to the parameter
- \(\alpha\): learning rate

The gradient indicates how the loss changes as the parameter changes. Subtracting the gradient moves the parameter in a local direction that reduces the loss when the step is suitable.

### Learning Rate

The learning rate controls the size of each update.

- Too small: learning may be slow.
- Too large: updates may overshoot and fail to converge.
- Suitable: updates can progressively reduce the loss.

### Types of Gradient Descent

- **Batch Gradient Descent:** Computes an update using the full training dataset.
- **Stochastic Gradient Descent (SGD):** Uses one training example per update.
- **Mini-batch Gradient Descent:** Uses a small batch of examples per update.

Mini-batch methods are widely used in modern deep learning.

---

## 13. Model Parameters vs Hyperparameters

| Model parameters | Hyperparameters |
|---|---|
| Learned from training data | Set before training or adjusted during model selection |
| Examples: Linear Regression weights and bias | Examples: learning rate, tree depth, KNN neighbor count |
| Directly define the fitted model | Influence how the model is built or trained |

Hyperparameters can be selected using validation data or cross-validation.

Do not repeatedly tune a model based on the test set. Doing so gradually leaks information from the test set into model selection.

---

## 14. Cross-Validation

Cross-validation is a method for evaluating how a model may perform across different subsets of data.

In \(k\)-fold cross-validation:

1. Divide the training data into \(k\) folds.
2. Train on \(k-1\) folds.
3. Evaluate on the remaining fold.
4. Repeat until each fold has been used for evaluation.
5. Combine the evaluation results.

For example, five-fold cross-validation performs five training and validation rounds.

Cross-validation is useful when data is limited, but it must respect the problem structure. Time-series data, grouped samples, and other dependent observations may require specialized splitting strategies.

The final test set should remain separate from the model-selection process.

---

## 15. Common Machine Learning Algorithms

Learn these in a practical order.

### Regression
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

### Classification
4. Logistic Regression
5. K-Nearest Neighbors (KNN)
6. Decision Tree Classifier
7. Random Forest Classifier
8. Support Vector Machine (SVM)

### Unsupervised Learning
9. K-Means Clustering
10. Principal Component Analysis (PCA)

This is a suggested learning order, not a universal ranking. Choose algorithms based on the task and compare them using appropriate evaluation methods.

---

## 16. A Basic ML Workflow in Python

Typical imports for a supervised learning workflow:

```python
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
```

Example using a dataset with numerical features:

```python
# Load data
df = pd.read_csv("data.csv")

# Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# Split before fitting preprocessing steps
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Numerical-only example
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

# Learn from training data
model.fit(X_train, y_train)

# Predict unseen test examples
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
print("MAE:", mae)
```

**Important:** This example assumes every column in `X` is numerical and the target is suitable for regression. Categorical features need appropriate encoding, and a classification problem needs a classifier and suitable classification metrics.

The file name `data.csv` and column name `target` are placeholders. Replace them with the actual dataset and target column.

---

## 17. Common Mistakes to Avoid

1. Training and evaluating on the same data.
2. Preprocessing the entire dataset before splitting.
3. Using the wrong algorithm for the task.
4. Measuring a model using only one unsuitable metric.
5. Tuning repeatedly on the test set.
6. Ignoring missing values and categorical columns.
7. Assuming a high training score guarantees good real-world performance.
8. Learning algorithms only by memorizing definitions.
9. Using complex models before establishing a simple baseline.
10. Deploying a model without checking its behavior on realistic inputs.

---

## 18. Recommended Learning Order

Follow this sequence:

- [ ] Understand ML terminology and learning types.
- [ ] Revise NumPy and Pandas.
- [ ] Learn basic statistics and probability as needed.
- [ ] Understand train, validation, and test splits.
- [ ] Practice missing-value handling, encoding, and scaling.
- [ ] Learn `ColumnTransformer` and `Pipeline`.
- [ ] Learn Linear Regression.
- [ ] Understand loss functions and gradient descent.
- [ ] Learn classification using Logistic Regression.
- [ ] Learn evaluation metrics.
- [ ] Study overfitting, underfitting, bias, and variance.
- [ ] Learn Decision Trees and Random Forests.
- [ ] Practice cross-validation and hyperparameter tuning.
- [ ] Build an end-to-end ML project.
- [ ] Learn model persistence and deployment.
- [ ] Move into more advanced topics, then Deep Learning.

---

## 19. Quick Revision

**Machine Learning:** Learning patterns from data.

**Feature:** Input used to make a prediction.

**Target:** Value the model predicts.

**Model:** Learned mapping from inputs to predictions.

**Training:** Learning model parameters.

**Inference:** Predicting with a trained model.

**Supervised learning:** Learns from labeled examples.

**Unsupervised learning:** Finds structure in unlabeled data.

**Reinforcement learning:** Learns through actions and rewards.

**Regression:** Predicts numerical values.

**Classification:** Predicts classes.

**Preprocessing:** Transforms data before modeling.

**Pipeline:** Chains preprocessing and model steps.

**Loss:** Quantifies prediction error for optimization.

**Gradient Descent:** Updates parameters to reduce loss.

**Overfitting:** Fits training data too closely and generalizes poorly.

**Underfitting:** Fails to learn important patterns.

**Generalization:** Performs well on unseen data.

**Data leakage:** Uses information that should not influence training or evaluation.

**Cross-validation:** Evaluates a model across multiple data splits.

**Parameters:** Learned values.

**Hyperparameters:** Settings that control learning or model complexity.

---

## 20. Self-Test

1. What is the difference between AI, ML, and Deep Learning?
2. What is the difference between a feature and a target?
3. Is predicting house prices regression or classification?
4. Is spam detection regression or classification?
5. Why do we split data before fitting preprocessing steps?
6. What is the difference between training and inference?
7. What causes overfitting?
8. What does the learning rate control?
9. What is the difference between model parameters and hyperparameters?
10. Why should we keep the test set separate from model tuning?
11. What is the purpose of a preprocessing pipeline?
12. Why can accuracy be misleading for an imbalanced classification dataset?


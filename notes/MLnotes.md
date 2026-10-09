# AI/ML Technical Terms — Complete Study Notes

## 1. Artificial Intelligence (AI)

**Definition:** Artificial Intelligence is the field of building systems that perform tasks associated with human intelligence, such as understanding language, recognizing images, reasoning, and making decisions.

### Major areas of AI
- **Machine Learning (ML):** Learning patterns from data.
- **Deep Learning (DL):** Machine Learning using neural networks with multiple layers.
- **NLP:** Working with human language.
- **Computer Vision:** Working with images and videos.
- **Robotics:** Building systems that sense and act in the physical world.
- **Generative AI:** Generating new content such as text, images, audio, video, and code.

Relationship:

```text
Artificial Intelligence
└── Machine Learning
    └── Deep Learning
        ├── Neural Networks
        ├── Transformers
        └── Many modern AI models
```

NLP and Computer Vision are application areas that can use both ML and DL.

---

## 2. Machine Learning (ML)

**Definition:** Machine Learning is a field in which computers learn patterns from data instead of relying entirely on explicitly programmed rules.

Traditional programming:

```text
Rules + Data → Program → Output
```

Machine Learning:

```text
Training Data → Learning Algorithm → Trained Model
```

Example: Instead of manually writing rules to identify spam emails, we train a model using examples of spam and legitimate emails.

---

## 3. Dataset

**Definition:** A dataset is a collection of data used for analysis, training, or evaluation.

Example:

| Age | Salary | City | Purchased |
|---:|---:|---|---|
| 20 | 20000 | Delhi | No |
| 30 | 50000 | Mumbai | Yes |
| 25 | 35000 | Delhi | Yes |

### Sample / Instance / Observation

One row of a dataset is called a sample, instance, or observation. These terms are often interchangeable.

### Feature

A feature is an input variable used by the model.

Example:
- Age
- Salary
- City

Other terms include input variable and, in many contexts, independent variable.

### Target / Label

The target is the value the model is supposed to predict.

Example: `Purchased`

Other terms include output variable and, in many contexts, dependent variable. The word *label* is especially common in classification.

### X and y

In scikit-learn, the convention is:

```python
X = df[["age", "salary", "city"]]
y = df["purchased"]
```

- `X` = input features, usually a 2D table.
- `y` = target values, usually a 1D series.

### Feature Matrix and Target Vector

- **Feature matrix:** The table of input features, represented by `X`.
- **Target vector:** The corresponding target values, represented by `y`.

---

## 4. Training, Validation, and Test Data

### Training Data

The data used to train the model.

Common variables:

```python
X_train
y_train
```

### Validation Data

Data used during model development to compare models, tune hyperparameters, and make modeling decisions.

### Test Data

Data kept separate to estimate the final model's performance on unseen examples.

The test set should not be repeatedly used to make modeling decisions.

### Train-Test Split

The process of dividing data into training and test sets.

Example:

```text
1000 samples
├── 800 training samples
└── 200 test samples
```

An 80/20 split is common, but the appropriate split depends on the dataset and problem.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Generalization

A model's ability to perform well on new, unseen data.

**Goal:** Learn useful patterns that generalize rather than simply memorize the training data.

---

## 5. Data Preprocessing

**Definition:** Data preprocessing means preparing raw data so it can be used reliably by an ML algorithm.

Major preprocessing operations:

- Encoding
- Scaling
- Imputation
- Feature engineering
- Feature selection

### Encoding

Encoding converts categorical information into a numerical representation.

Example: One-Hot Encoding

```text
City      Delhi  Mumbai
Delhi       1      0
Mumbai      0      1
```

`OneHotEncoder` is a scikit-learn transformer that learns categories from training data and converts categorical columns into numerical features.

### Scaling

Scaling changes the numerical scale or range of features.

Example:

```text
Age:       18–60
Salary:    20,000–200,000
```

Some algorithms are sensitive to these differences in scale.

#### Standardization

`StandardScaler` uses the formula:

\[
z = \frac{x-\mu}{\sigma}
\]

Where:
- \(x\) = original value
- \(\mu\) = mean
- \(\sigma\) = standard deviation
- \(z\) = standardized value

After standardization, the training feature typically has mean approximately 0 and standard deviation approximately 1.

#### Min-Max Scaling

`MinMaxScaler` typically maps values into a fixed range, commonly 0 to 1.

Formula:

\[
x' = \frac{x-x_{\min}}{x_{\max}-x_{\min}}
\]

Future values can fall outside the chosen range if they exceed the training data's minimum or maximum.

**Note:** The word *normalization* is used differently in different contexts. Learn the exact transformation being used.

### Imputation

Imputation means filling in missing values.

Example:

```text
Age
20
25
NaN
30
```

The missing value could be replaced with a suitable statistic, such as the median calculated from the training data.

Common strategies:
- Mean imputation
- Median imputation
- Most frequent value
- Constant value

### Feature Engineering

Feature engineering means creating useful features from existing data.

Examples:
- Date of birth → Age
- Quantity × Unit price → Total price
- Height and weight → BMI

### Feature Selection

Feature selection means selecting useful features and removing irrelevant, redundant, or potentially harmful ones.

Example:

```text
Features:
Age
Salary
City
Random ID
```

The random ID might not provide useful predictive information and may be excluded.

### Dimensionality

Dimensionality generally refers to the number of features in a representation.

A dataset with five input features has five feature dimensions.

### Dimensionality Reduction

Dimensionality reduction reduces the number of dimensions while attempting to preserve useful information.

**PCA (Principal Component Analysis)** is a common dimensionality-reduction technique.

---

## 6. Preprocessor

**Definition:** A preprocessor is a component or collection of transformations that prepares data before it reaches the ML model.

Example:

```text
Raw Data
   |
   ├── Numeric columns → StandardScaler
   |
   └── Categorical columns → OneHotEncoder
                     |
                     v
              Processed Data
```

A preprocessor does not usually make predictions. Its job is to transform the input data.

### ColumnTransformer

`ColumnTransformer` is a scikit-learn tool that applies different transformations to different columns.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["age", "salary"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["city"])
    ]
)
```

Explanation:

```python
("num", StandardScaler(), ["age", "salary"])
```

- `"num"` = name of the transformation.
- `StandardScaler()` = transformer.
- `["age", "salary"]` = columns to transform.

Similarly:

```python
("cat", OneHotEncoder(handle_unknown="ignore"), ["city"])
```

This applies One-Hot Encoding to `city`.

The transformed outputs are combined into one feature representation.

### fit()

`fit()` learns the information needed for preprocessing or training.

Examples:
- `StandardScaler` learns the mean and standard deviation.
- `OneHotEncoder` learns the categories present in the supplied training data.
- A predictive model learns its parameters.

```python
preprocessor.fit(X_train)
```

### transform()

`transform()` applies previously learned transformation rules to data.

```python
X_train_processed = preprocessor.transform(X_train)
```

### fit_transform()

`fit_transform()` fits the transformer and transforms the supplied data.

```python
X_train_processed = preprocessor.fit_transform(X_train)
```

### Correct Preprocessing Workflow

```text
Training Data
     |
     v
fit_transform()
     |
     ├── Learn statistics/categories
     └── Transform training features

Test Data
     |
     v
transform()
     |
     └── Apply the same learned rules
```

**Important:** Fit preprocessing on training data only. Apply the learned transformations to validation and test data.

This helps prevent data leakage.

---

## 7. Pipeline

**Definition:** A Pipeline connects multiple processing and modeling steps into one workflow.

Example:

```text
Raw Data
   |
   v
Preprocessor
   |
   v
Processed Features
   |
   v
ML Model
   |
   v
Prediction
```

In scikit-learn:

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
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

### Difference Between Preprocessor and Pipeline

- **Preprocessor:** Prepares data.
- **Model:** Learns patterns and makes predictions.
- **Pipeline:** Connects preprocessing and model steps.

A pipeline applies the fitted preprocessing steps automatically before making predictions.

It also helps prevent data leakage when used correctly with training splits and cross-validation.

---

## 8. Algorithm and Model

### Algorithm

An algorithm is a method used to learn patterns from data.

Examples:
- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- K-Means

### Model

A model is the learning system. More precisely, a trained model is the fitted result containing learned information that can make predictions.

### Training

Training means fitting a model to data.

```python
model.fit(X_train, y_train)
```

### Parameter

A parameter is a value learned from training data.

For Linear Regression:

\[
y = wx+b
\]

- \(w\) = weight/coefficient
- \(b\) = bias/intercept

These are learned parameters in this example.

### Hyperparameter

A hyperparameter is a configuration selected before or during training rather than learned as an ordinary model parameter.

Example:

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=5
)
```

- `n_estimators` = number of trees.
- `max_depth` = maximum tree depth.

Both are hyperparameters.

**Remember:**

```text
Parameters → Learned from training data
Hyperparameters → Configured by us
```

### Prediction

Prediction is the output produced by a trained model.

```python
predictions = model.predict(X_test)
```

### Inference

Inference means using a trained model to produce outputs for new inputs.

Training = learning.

Inference = using what was learned.

---

## 9. Supervised, Unsupervised, and Reinforcement Learning

### Supervised Learning

The training data includes input features and known target values.

```text
Features + Known Targets
           |
           v
      ML Algorithm
           |
           v
       Trained Model
```

Examples:
- Predict house prices.
- Classify spam emails.
- Predict customer churn.

### Unsupervised Learning

The data has no provided target labels. The algorithm searches for structure or patterns.

Examples:
- Customer clustering
- Dimensionality reduction
- Some types of anomaly detection

### Reinforcement Learning

An agent learns to choose actions by interacting with an environment and receiving rewards or penalties.

Important terms:
- **Agent:** The learner or decision-maker.
- **Environment:** The world the agent interacts with.
- **Action:** A choice made by the agent.
- **Reward:** Feedback about an outcome.
- **Policy:** The strategy used to choose actions.

### Clustering

Clustering groups similar examples without using supplied class labels.

Example: K-Means groups customers according to their features.

The algorithm creates groups, but humans may need to interpret what those groups mean.

---

## 10. Regression and Classification

### Regression

Regression predicts a numerical quantity, often a continuous value.

Examples:
- House price
- Temperature
- Sales
- Salary

Example:

```text
Input: House features
Output: Predicted price = ₹52,50,000
```

### Classification

Classification predicts a category or class.

Examples:
- Spam / Not Spam
- Cat / Dog
- Fraud / Not Fraud

### Binary Classification

A classification task with two classes.

Examples:
- Yes / No
- 0 / 1
- Spam / Not Spam

### Multiclass Classification

A classification task with more than two possible classes, where each example is assigned one class.

Example:

```text
Cat
Dog
Horse
```

### Multilabel Classification

A task where one example can have multiple labels simultaneously.

Example: An image may contain a car, a person, and a dog.

---

## 11. Loss Function and Optimization

### Loss Function

A loss function measures prediction error according to a chosen mathematical rule.

Examples:
- Mean Squared Error (MSE)
- Cross-Entropy Loss

A lower loss generally means better predictions according to that particular loss function.

### Cost Function / Objective Function

The function that an optimization procedure tries to minimize or maximize.

Terminology varies by context. An objective may include an aggregate loss and regularization terms.

### Optimization

Optimization means adjusting model parameters to improve the objective.

### Gradient

A gradient describes how a function changes with respect to its parameters.

In neural-network training, gradients help determine how parameters should change to reduce the loss.

### Gradient Descent

Gradient Descent is an optimization method that repeatedly updates parameters in a direction intended to reduce the objective.

```text
Calculate Prediction
        |
        v
Calculate Loss
        |
        v
Calculate Gradient
        |
        v
Update Parameters
        |
        v
Repeat
```

### Learning Rate

The learning rate controls the size of parameter updates in many optimization methods.

- Too high → training may become unstable or overshoot.
- Too low → training may be very slow.
- Appropriate → updates can make steady progress.

### Epoch

One complete pass through the training dataset.

Example:

```text
1000 training samples

1 epoch  → one complete pass
10 epochs → ten complete passes
```

The term is especially common in neural-network training.

### Batch

A subset of training samples processed together.

### Batch Size

The number of examples processed in one batch.

Example:

```text
1000 samples
Batch size = 100
Number of batches = 10
```

This assumes all samples are processed and the final batch is not dropped.

---

## 12. Overfitting and Underfitting

### Overfitting

Overfitting happens when a model learns the training data too specifically, including noise or accidental patterns.

Example:

```text
Training performance: 99%
Test performance:     65%
```

The model performs well on training data but poorly on unseen data.

### Underfitting

Underfitting happens when a model is too limited to learn important patterns.

Example:

```text
Training performance: 65%
Test performance:     63%
```

Both performances are poor.

### Bias

In the bias-variance context, bias refers to error associated with overly restrictive assumptions.

High bias is often associated with underfitting.

### Variance

Variance describes how sensitive a model is to the particular training data it receives.

High variance is often associated with overfitting.

### Bias-Variance Trade-Off

A good model needs enough flexibility to learn useful patterns without becoming too sensitive to the training examples.

### Regularization

Regularization discourages overly complex models or large parameter values to help reduce overfitting.

Examples:
- L1 regularization
- L2 regularization

---

## 13. Evaluation Metrics

Evaluation metrics summarize how well a model performs.

The correct metric depends on the problem and the consequences of errors.

### Accuracy

The proportion of predictions that are correct.

\[
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}
\]

Accuracy can be misleading with imbalanced classes.

Example:

```text
990 negative examples
10 positive examples
```

A model that always predicts negative achieves 99% accuracy but fails to detect any positive examples.

### Confusion Matrix

A table comparing actual classes with predicted classes.

Four important terms:

- **TP — True Positive:** Actual positive, predicted positive.
- **TN — True Negative:** Actual negative, predicted negative.
- **FP — False Positive:** Actual negative, predicted positive.
- **FN — False Negative:** Actual positive, predicted negative.

### Precision

Of all examples predicted positive, how many were actually positive?

\[
Precision = \frac{TP}{TP+FP}
\]

Useful when false positives are costly.

### Recall

Of all actual positive examples, how many did the model correctly identify?

\[
Recall = \frac{TP}{TP+FN}
\]

Useful when false negatives are costly.

### F1 Score

The harmonic mean of precision and recall.

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision+Recall}
\]

It combines precision and recall into one score.

### ROC-AUC

ROC-AUC summarizes how well a classifier ranks positive examples above negative examples across different thresholds.

For highly imbalanced datasets, precision-recall analysis can also be especially informative.

### Regression Metrics

**MAE — Mean Absolute Error**

The average absolute difference between predictions and actual values.

**MSE — Mean Squared Error**

The average squared prediction error. Larger errors receive a stronger penalty.

**RMSE — Root Mean Squared Error**

The square root of MSE. It is expressed in the same units as the target.

**R² — Coefficient of Determination**

Compares model performance with a baseline that predicts the target mean. R² can be negative on test data.

---

## 14. Cross-Validation

**Definition:** Cross-validation estimates model performance by repeatedly fitting and evaluating the model on different partitions of the development data.

### 5-Fold Cross-Validation

The data is divided into five folds.

```text
Round 1: Fold 1 = validation
Round 2: Fold 2 = validation
Round 3: Fold 3 = validation
Round 4: Fold 4 = validation
Round 5: Fold 5 = validation
```

Each fold gets a turn as validation data while the remaining folds are used for training.

Cross-validation is useful for model comparison and hyperparameter tuning.

Keep the final test set separate from these development decisions.

---

## 15. Data Leakage

**Definition:** Data leakage happens when information that should not be available during training or evaluation influences the model.

Examples:
- Fitting a scaler on the entire dataset before splitting it.
- Using test-set information to choose model settings.
- Including a feature that directly reveals the target.

Correct workflow:

```text
Split Dataset
     |
     v
Fit Preprocessor on Training Data
     |
     v
Train Model
     |
     v
Transform Test Data Using Fitted Preprocessor
     |
     v
Evaluate Model
```

Using a scikit-learn Pipeline helps reduce common preprocessing leakage mistakes.

---

## 16. Neural Networks and Deep Learning

### Neural Network

A model consisting of connected computational units organized into layers.

```text
Input Layer
     |
     v
Hidden Layer(s)
     |
     v
Output Layer
```

### Activation Function

An activation function introduces nonlinear behavior into a neural network.

Common examples:

- **ReLU:** Commonly used in hidden layers.
- **Sigmoid:** Maps values to the range 0–1.
- **Tanh:** Maps values to the range -1 to 1.
- **Softmax:** Converts a vector of scores into values that sum to 1, commonly used for multiclass outputs.

### Forward Pass

The process of passing input through a neural network to calculate its output.

### Backpropagation

A method used to calculate gradients through a neural network so that its parameters can be updated during training.

Simplified process:

```text
Input
  |
  v
Forward Pass
  |
  v
Prediction
  |
  v
Loss
  |
  v
Backpropagation
  |
  v
Gradients
  |
  v
Parameter Updates
```

### CNN — Convolutional Neural Network

A neural-network architecture commonly used for image-related tasks.

### RNN — Recurrent Neural Network

A family of architectures designed to process sequences. They remain useful in some contexts, although Transformers are prominent in modern language systems.

### Transformer

A neural-network architecture built around attention mechanisms.

Transformers are central to many modern language models and are also used for other types of data.

### Attention

A mechanism that allows a model to weigh the relevance of different parts of its input when building a representation.

---

## 17. NLP, Generative AI, and LLMs

### NLP — Natural Language Processing

The field of AI focused on human language.

Examples:
- Sentiment analysis
- Machine translation
- Text classification
- Summarization
- Question answering

### Generative AI

AI systems that generate new content such as text, images, audio, video, and code.

### LLM — Large Language Model

A large neural network trained to model language and generate or process text.

Applications:
- Writing
- Summarization
- Coding assistance
- Question answering
- Text analysis

LLM outputs can be incorrect, so they need verification.

### Token

A token is a unit of text processed by a language model.

A token may represent:
- A whole word
- Part of a word
- Punctuation
- Another text unit

One word does not necessarily equal one token.

### Tokenization

The process of converting text into tokens and usually token IDs that a model can process.

### Vector

A vector is an ordered collection of numbers.

Example:

```text
[0.2, -0.4, 0.7, 0.1]
```

Vectors can represent features, embeddings, parameters, or gradients.

### Embedding

An embedding is a numerical vector representation of an item, such as text, intended to capture useful properties.

Embeddings are important for:
- Semantic search
- Recommendation systems
- Clustering
- Retrieval-Augmented Generation

### Similarity Search

Similarity search finds items whose vector representations are close to a query vector under a chosen similarity or distance measure.

It can retrieve related meanings rather than relying only on exact keyword matches.

### Vector Database

A system designed to store and search vector representations, often alongside metadata and associated documents.

### RAG — Retrieval-Augmented Generation

RAG combines information retrieval with text generation.

Simplified workflow:

```text
User Question
     |
     v
Retrieve Relevant Document Chunks
     |
     v
Provide Context + Question to LLM
     |
     v
Generate Answer
```

RAG can help a model answer using specific external documents, but retrieval and generation can still fail.

### Fine-Tuning

Further training a pre-trained model on a selected dataset to adapt its behavior or capabilities.

Fine-tuning is different from RAG:
- **RAG:** Retrieves information and provides it as context.
- **Fine-tuning:** Updates model parameters through additional training.

---

## 18. Deployment and Backend Terms

### Deployment

Making a trained model available for practical use in an application or service.

### API — Application Programming Interface

A defined way for software components to communicate.

A web API can accept input from a frontend and return a prediction from a backend.

### Inference Endpoint

An API route or service that accepts input and returns a model's output.

### FastAPI

A Python framework for building web APIs. It can expose a trained ML model through HTTP endpoints.

### Docker / Container

Docker packages an application and its dependencies into a container image.

A running container provides an isolated environment for the application.

Containers help make deployments more reproducible but do not automatically guarantee identical behavior across every environment.

Typical deployment flow:

```text
Trained Model + Preprocessing
             |
             v
          FastAPI
             |
             v
           Docker
             |
             v
        Server / Cloud
             |
             v
        Frontend / Users
```

---

## 19. Essential scikit-learn Methods

| Method | Meaning |
|---|---|
| `fit(X, y)` | Fits an estimator to data. Exact behavior depends on the estimator. |
| `transform(X)` | Applies a fitted transformation. |
| `fit_transform(X)` | Fits a transformer and transforms the supplied data. |
| `predict(X)` | Produces predictions from a fitted estimator. |
| `predict_proba(X)` | Estimates class probabilities when supported. |
| `score(X, y)` | Returns an estimator-specific score; check which score is used. |

Note:
- Transformers commonly provide `fit()` and `transform()`.
- Predictive estimators commonly provide `fit()` and `predict()`.
- Not every object supports every method.

---

## 20. Priority List: What to Learn First

Do not try to master every term simultaneously.

### Priority 1 — Current ML Foundation

- Dataset, sample, feature, target
- `X` and `y`
- Training, validation, and test data
- Train-test split
- Preprocessing
- Encoding, scaling, and imputation
- Feature engineering and feature selection
- `ColumnTransformer`
- Preprocessor and Pipeline
- `fit()`, `transform()`, `fit_transform()`, `predict()`
- Algorithm and trained model
- Parameters and hyperparameters
- Regression and classification
- Overfitting and underfitting
- Generalization and data leakage

### Priority 2 — Model Evaluation and Improvement

- Loss function
- Optimization and Gradient Descent
- Cross-validation
- Confusion matrix
- TP, TN, FP, FN
- Accuracy, precision, recall, F1, ROC-AUC
- MAE, MSE, RMSE, R²
- Bias and variance
- Regularization
- Hyperparameter tuning

### Priority 3 — Deep Learning

- Neural networks
- Layers and activation functions
- Forward pass
- Gradients and backpropagation
- Epoch, batch, and batch size
- Learning rate
- CNN, RNN, Transformer, and attention

### Priority 4 — Generative AI and Deployment

- NLP and LLMs
- Tokens and tokenization
- Embeddings and vectors
- Similarity search
- Vector databases
- RAG and fine-tuning
- APIs and inference
- FastAPI, Docker, and deployment

---

## 21. My Current Learning Roadmap

```text
Python
  |
  v
NumPy
  |
  v
Pandas
  |
  v
Matplotlib
  |
  v
Data Preprocessing
  ├── Encoding
  ├── Scaling
  └── Preprocessor / ColumnTransformer  <-- Current Focus
             |
             v
          Pipeline
             |
             v
    Regression / Classification
             |
             v
      Model Training
             |
             v
       Evaluation
             |
             v
   Overfitting / Model Tuning
             |
             v
         ML Projects
             |
             v
        Deep Learning
             |
             v
    AI Engineering / LLMs
```

## 22. How to Study Every New Technical Term

For each new concept, write down:

1. **Definition:** What is it?
2. **Purpose:** Why do we need it?
3. **Working:** What happens internally?
4. **Example:** A small code or data example.
5. **Workflow:** Where does it fit into the ML process?

### Final Reminder

Do not learn technical terms just to memorize their names. Understand the concept, implement it in code, and learn when to use it.

Strong ML foundations are more valuable than jumping directly into advanced terminology.

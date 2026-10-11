# 11. Naive Bayes

## 1. What Is Naive Bayes?

Naive Bayes is a supervised machine learning algorithm used mainly for classification. It predicts a class by calculating probabilities using Bayes' theorem.

**Applications:**
- Email spam detection
- Text classification
- Sentiment analysis
- Document categorization

Unlike KNN, which predicts using nearby training examples, Naive Bayes uses probabilities learned from the training data.

## 2. Bayes' Theorem

Formula:

P(A | B) = P(B | A) × P(A) / P(B)

Where:

- P(A | B): probability of A given B.
- P(B | A): probability of B given A.
- P(A): prior probability of A.
- P(B): overall probability of B.

### Spam example

Suppose:

- 40% of emails are spam.
- 70% of spam emails contain the word "prize".
- 10% of non-spam emails contain "prize".

Therefore:

P(Spam) = 0.40

P(Prize | Spam) = 0.70

P(Prize | Not Spam) = 0.10

First, calculate the overall probability of "prize":

P(Prize) = (0.70 × 0.40) + (0.10 × 0.60)

P(Prize) = 0.28 + 0.06 = 0.34

Then:

P(Spam | Prize) = (0.70 × 0.40) / 0.34

P(Spam | Prize) ≈ 0.8235 = 82.35%

**Important distinction:**

- P(Prize | Spam): probability of "prize" appearing in a spam email.
- P(Spam | Prize): probability that an email is spam given that it contains "prize".

These are different probabilities.

## 3. The Naive Independence Assumption

Naive Bayes assumes that features are conditionally independent given the class.

For example, in spam detection:

- x₁ = contains "win"
- x₂ = contains "prize"
- x₃ = contains "free"

The model assumes that these features are independent of one another once the class is known.

This assumption is often not perfectly true in real data, but the model can still perform well.

Without the assumption, we need the joint conditional probability:

P(x₁, x₂, x₃ | C)

With the assumption:

P(x₁, x₂, x₃ | C) ≈ P(x₁ | C) × P(x₂ | C) × P(x₃ | C)

Here, C represents a class, such as Spam.

## 4. How Naive Bayes Makes Predictions

For a class C and features x₁, x₂, ..., xₙ, the class score is proportional to:

Score(C) = P(C) × P(x₁ | C) × ... × P(xₙ | C)

The model:

1. Learns class priors from training data.
2. Estimates feature probabilities for each class.
3. Calculates a score for each possible class.
4. Predicts the class with the highest score.

The scores can be normalized to obtain probabilities that sum to 1.

## 5. Numerical Dry Run

Suppose a new email contains "win" and "prize".

| Probability | Spam | Not Spam |
|---|---:|---:|
| Prior probability | 0.40 | 0.60 |
| P(win | class) | 0.70 | 0.10 |
| P(prize | class) | 0.80 | 0.20 |

Spam score:

0.40 × 0.70 × 0.80 = 0.224

Not Spam score:

0.60 × 0.10 × 0.20 = 0.012

Spam has the higher score, so the prediction is Spam.

Normalize the scores:

P(Spam | features) ≈ 0.224 / (0.224 + 0.012)

P(Spam | features) ≈ 94.92%

These are illustrative probabilities based on the assumed feature independence.

## 6. Laplace Smoothing

A feature that never appears in a class may receive a probability of zero.

Because Naive Bayes multiplies probabilities, a single zero can make the entire class score zero.

Laplace smoothing prevents this by adding a small count to feature counts.

For Multinomial Naive Bayes:

P(word | class) = (count(word, class) + α) / (total word count in class + αV)

Where:

- count(word, class): number of occurrences of the word in the class.
- total word count in class: total word occurrences for that class.
- V: vocabulary size.
- α: smoothing parameter, commonly 1.

Smoothing prevents zero probabilities for words absent from a class's training examples.

## 7. Main Types of Naive Bayes

### GaussianNB

- Models continuous numerical features using a Gaussian (normal) distribution within each class.
- Example: flower classification using petal and sepal measurements.
- scikit-learn class: `GaussianNB`.

### MultinomialNB

- Designed for count-like features.
- Commonly used for word counts in text classification.
- scikit-learn class: `MultinomialNB`.

### BernoulliNB

- Designed for binary features.
- Useful when features represent whether a word or property is present or absent.
- scikit-learn class: `BernoulliNB`.

Choose a variant based on the feature representation and the assumptions that suit the dataset.

## 8. Implementing Multinomial Naive Bayes

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

X_train = [
    "win money now",
    "claim your free prize",
    "free money prize",
    "win a free gift",
    "team meeting today",
    "project meeting today",
    "schedule the team meeting",
    "discuss project work",
]

y_train = [
    "spam",
    "spam",
    "spam",
    "spam",
    "not spam",
    "not spam",
    "not spam",
    "not spam",
]

# Learn the vocabulary and convert text to word counts.
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)

# Train the model.
model = MultinomialNB(alpha=1.0)
model.fit(X_train_counts, y_train)

# Predict a new email.
email = ["claim your prize now"]
email_counts = vectorizer.transform(email)

prediction = model.predict(email_counts)
probabilities = model.predict_proba(email_counts)

print("Prediction:", prediction[0])

for label, probability in zip(model.classes_, probabilities[0]):
    print(f"{label}: {probability:.2%}")
```

### Understanding the workflow

- `CountVectorizer`: converts text into numerical word counts.
- `fit_transform()`: learns the vocabulary from training text and transforms that text.
- `MultinomialNB(alpha=1.0)`: creates a Multinomial Naive Bayes classifier with Laplace smoothing.
- `fit()`: learns from the training features and labels.
- `transform()`: converts new text using the existing vocabulary.
- `predict()`: predicts the class.
- `predict_proba()`: returns estimated class probabilities.

Use `transform()` on new or held-out text instead of fitting a new vocabulary. This preserves the feature representation learned from training data.

## 9. Train/Test Split and Model Evaluation

A model should be evaluated on data that was not used to train it.

Example using GaussianNB and the Iris dataset:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))
```

### Evaluation metrics

- **Accuracy:** proportion of predictions that are correct.
- **Precision:** among predicted positive examples, how many are actually positive?
- **Recall:** among actual positive examples, how many were identified?
- **F1-score:** harmonic mean of precision and recall.
- **Confusion matrix:** summarizes correct and incorrect predictions by class.

Accuracy alone can be misleading for imbalanced datasets.

## 10. Strengths

- Simple and fast to train.
- Efficient for high-dimensional text features.
- Can work well with relatively small training datasets.
- Supports probabilistic classification.
- Provides several variants for different types of data.

## 11. Limitations

- The conditional independence assumption may be unrealistic.
- Correlated features can affect performance.
- Probability estimates may not be well calibrated.
- Zero probabilities require attention, especially for unseen words.
- Performance depends on choosing an appropriate variant and feature representation.

## 12. Common Mistakes

1. Confusing P(A | B) with P(B | A).
2. Assuming all features are truly independent in real life.
3. Using a model variant that does not suit the features.
4. Fitting a new text vocabulary on test data.
5. Evaluating only on training data.
6. Trusting accuracy alone for imbalanced classes.
7. Treating predicted probabilities as guaranteed certainty.

## 13. Quick Revision

- Naive Bayes is a probability-based classification algorithm.
- It uses Bayes' theorem.
- Its defining assumption is conditional independence of features given the class.
- Class scores combine prior probabilities with feature likelihoods.
- Laplace smoothing prevents zero feature probabilities.
- GaussianNB suits continuous features modeled with Gaussian distributions.
- MultinomialNB suits count-like features.
- BernoulliNB suits binary features.
- Use a held-out test set to evaluate generalization.
- Inspect precision, recall, F1-score, and the confusion matrix when appropriate.

## 14. Self-Test

1. What is the difference between P(Spam | Prize) and P(Prize | Spam)?
2. What is the conditional independence assumption?
3. Why does Naive Bayes use Laplace smoothing?
4. Which variant is suitable for continuous flower measurements?
5. Which variant is commonly used for word counts?
6. Why should we use `transform()` on held-out text?
7. Why can accuracy be misleading for imbalanced classes?

# AI/ML ENGINEER — MASTER LEARNING ROADMAP

```text
AI/ML ENGINEER
│
├── 00. DEVELOPMENT SETUP
│   ├── Python Environment & Virtual Environments
│   ├── VS Code, Jupyter & Packages
│   ├── Git & GitHub
│   └── Project Structure, Testing & Debugging
│
├── 01. PYTHON & DATA FOUNDATIONS
│   ├── Python Revision
│   │   ├── Functions, Classes, Modules & Exceptions
│   │   └── Iterators, Generators & Comprehensions
│   ├── NumPy
│   │   ├── Arrays, Shapes, Indexing & Broadcasting
│   │   └── Vectorization & Matrix Operations
│   ├── Pandas
│   │   ├── DataFrames, Filtering, Grouping & Merging
│   │   └── Missing Values & Data Cleaning
│   ├── Matplotlib & Data Visualization
│   └── Exploratory Data Analysis (EDA)
│
├── 02. MATHEMATICS FOR MACHINE LEARNING
│   ├── 2.1 Algebra & Functions
│   │   ├── Equations, Functions & Graphs
│   │   └── Exponents, Logarithms & Summations
│   ├── 2.2 Linear Algebra
│   │   ├── Scalars, Vectors, Matrices & Tensors
│   │   ├── Dot Products, Norms & Distances
│   │   ├── Matrix Multiplication & Linear Transformations
│   │   ├── Rank, Basis & Linear Independence
│   │   ├── Orthogonality & Projections
│   │   └── Eigenvalues, Eigenvectors & SVD
│   ├── 2.3 Calculus
│   │   ├── Limits & Derivatives
│   │   ├── Chain Rule & Partial Derivatives
│   │   ├── Gradients, Jacobians & Hessians
│   │   └── Computational Graphs
│   ├── 2.4 Probability
│   │   ├── Conditional Probability & Bayes' Theorem
│   │   ├── Random Variables & Distributions
│   │   ├── Expectation, Variance & Covariance
│   │   └── Joint, Marginal & Conditional Distributions
│   ├── 2.5 Statistics
│   │   ├── Descriptive Statistics & Sampling
│   │   ├── Estimation & Confidence Intervals
│   │   ├── Hypothesis Testing
│   │   └── Likelihood & Maximum Likelihood Estimation
│   └── 2.6 Optimization & Numerical Methods
│       ├── Objective Functions & Convexity
│       ├── Numerical Precision & Stability
│       └── Convergence & Regularization
│
├── 03. MACHINE LEARNING FUNDAMENTALS
│   ├── What Is Machine Learning?
│   ├── Supervised, Unsupervised & Reinforcement Learning
│   ├── Samples, Features, Labels & Targets
│   ├── Training, Validation & Test Sets
│   ├── Parameters vs Hyperparameters
│   ├── Predictions, Residuals & Errors
│   ├── Loss Functions vs Objective Functions
│   ├── Empirical Risk Minimization
│   ├── Gradients & Gradient Descent
│   ├── Learning Rate, Epochs, Batches & Iterations
│   ├── SGD, Momentum, RMSProp & Adam
│   ├── Overfitting, Underfitting & Generalization
│   ├── Bias–Variance Trade-off
│   └── Regularization & Model Complexity
│
├── 04. DATA PREPROCESSING & FEATURE ENGINEERING
│   ├── Data Inspection & Quality Checks
│   ├── Missing-Value Imputation
│   ├── Categorical Encoding
│   ├── Standardization & Normalization
│   ├── Outlier Handling
│   ├── Feature Engineering & Selection
│   ├── fit(), transform() & fit_transform()
│   ├── Scikit-learn Pipeline
│   ├── ColumnTransformer
│   ├── Data Leakage Prevention
│   └── Training–Inference Consistency
│
├── 05. SUPERVISED MACHINE LEARNING
│   ├── 5.1 Regression
│   │   ├── Simple & Multiple Linear Regression
│   │   ├── Least Squares & Polynomial Regression
│   │   ├── MAE, MSE & RMSE
│   │   └── Ridge, Lasso & Elastic Net
│   ├── 5.2 Classification
│   │   ├── Logistic Regression
│   │   ├── Sigmoid & Softmax
│   │   └── Cross-Entropy Loss
│   ├── 5.3 Distance & Probabilistic Models
│   │   ├── K-Nearest Neighbors (KNN)
│   │   └── Naive Bayes
│   ├── 5.4 Margin-Based Models
│   │   └── Support Vector Machines (SVM)
│   ├── 5.5 Tree-Based Models
│   │   ├── Decision Trees
│   │   ├── Entropy, Gini Impurity & Information Gain
│   │   └── Pruning
│   └── 5.6 Ensemble Learning
│       ├── Bagging & Random Forests
│       ├── Boosting & Gradient Boosting
│       └── XGBoost, LightGBM & CatBoost
│
├── 06. UNSUPERVISED MACHINE LEARNING
│   ├── Similarity & Distance Measures
│   ├── K-Means Clustering
│   ├── Hierarchical Clustering
│   ├── DBSCAN
│   ├── Gaussian Mixture Models & EM
│   ├── Dimensionality Reduction
│   ├── PCA & SVD Applications
│   └── Anomaly Detection
│
├── 07. MODEL EVALUATION & SELECTION
│   ├── Baselines & Data Splitting
│   ├── Cross-Validation & Stratification
│   ├── Confusion Matrix
│   ├── Accuracy, Precision, Recall & F1
│   ├── ROC-AUC & Precision–Recall Curves
│   ├── Regression Metrics & R²
│   ├── Class Imbalance & Threshold Selection
│   ├── Calibration & Error Analysis
│   ├── Hyperparameter Tuning
│   └── Reproducibility & Experiment Tracking
│
├── 08. END-TO-END CLASSICAL ML PROJECT
│   ├── Problem Definition & Success Metrics
│   ├── Dataset Collection & EDA
│   ├── Baseline Model
│   ├── Preprocessing & Training Pipeline
│   ├── Model Training & Comparison
│   ├── Hyperparameter Tuning
│   ├── Error Analysis & Evaluation
│   └── Model Saving & Documentation
│
├── 09. DEEP LEARNING FOUNDATIONS
│   ├── Perceptrons, Weights & Biases
│   ├── Activation Functions
│   ├── Forward Propagation
│   ├── Neural Network Loss Functions
│   ├── Computational Graphs & Autograd
│   ├── Backpropagation & Chain Rule
│   ├── Neural Network Optimization
│   ├── Weight Initialization
│   ├── Vanishing & Exploding Gradients
│   ├── Batch & Layer Normalization
│   └── Dropout, Weight Decay & Early Stopping
│
├── 10. PYTORCH & PRACTICAL DEEP LEARNING
│   ├── Tensors, Shapes, Devices & GPUs
│   ├── Datasets & DataLoaders
│   ├── torch.nn & Model Building
│   ├── Autograd & Gradient Inspection
│   ├── Training & Validation Loops
│   ├── Optimizers & Learning-Rate Schedulers
│   ├── Checkpoints & Model Serialization
│   └── Debugging & Performance Optimization
│
├── 11. DEEP LEARNING SPECIALIZATIONS
│   ├── Computer Vision
│   │   ├── Convolutions, Pooling & CNNs
│   │   ├── Transfer Learning & Augmentation
│   │   ├── Detection & Segmentation
│   │   └── Vision Transformers
│   ├── Sequential Models
│   │   ├── RNNs, LSTMs & GRUs
│   │   └── Time-Series Forecasting
│   └── Advanced Learning
│       ├── Self-Supervised Learning
│       ├── Contrastive Learning
│       └── Representation Learning
│
├── 12. NLP, TRANSFORMERS & LLMs
│   ├── Text Processing & Tokenization
│   ├── Bag-of-Words, TF-IDF & Embeddings
│   ├── Attention: Query, Key & Value
│   ├── Scaled Dot-Product Attention
│   ├── Multi-Head Attention
│   ├── Transformer Blocks & Positional Information
│   ├── Encoder, Decoder & Decoder-Only Models
│   ├── Next-Token Prediction & Pretraining
│   ├── Instruction Tuning & Fine-Tuning
│   ├── Hugging Face Transformers
│   ├── Parameter-Efficient Fine-Tuning
│   └── Quantization & Efficient Inference
│
├── 13. AI ENGINEERING & PRODUCTION
│   ├── Model Serving & FastAPI
│   ├── Input Validation & Output Schemas
│   ├── Docker & Deployment
│   ├── Testing, Logging & Monitoring
│   ├── CI/CD & Versioning
│   ├── Latency, Throughput & Cost
│   └── Model Drift, Reliability & Rollback
│
├── 14. RAG & LLM APPLICATIONS
│   ├── Document Ingestion & Parsing
│   ├── Chunking & Metadata
│   ├── Embeddings & Vector Search
│   ├── Sparse, Dense & Hybrid Retrieval
│   ├── Reranking & Retrieval Evaluation
│   ├── Context Construction & Generation
│   ├── Grounded Answers & Hallucination Evaluation
│   ├── Query Rewriting & Relevance Filtering
│   ├── Tool Calling & Agent Workflows
│   └── Security, Observability & Cost Optimization
│
└── 15. ADVANCED PROFESSIONAL DEPTH
    ├── Statistical Learning Theory
    ├── Bayesian ML & Uncertainty
    ├── Advanced Optimization & Numerical Stability
    ├── Distributed & Mixed-Precision Training
    ├── Model Compression & Quantization
    ├── Research Papers & Reproduction
    ├── Ablation Studies & Rigorous Evaluation
    ├── Responsible AI, Privacy & Robustness
    └── AI System Design & Production Trade-offs
```

## How we will follow this tree

1. We follow the numbered parts in order.
2. Each subtopic gets its own lesson, with mathematics, code, examples and a checkpoint where appropriate.
3. We mark each topic **Not started**, **Learning**, **Practiced**, or **Mastered**. Watching a lecture alone does not count as mastery.
4. We maintain this roadmap and progress in your `machine_learning/roadmap.md` file.
5. If a conversation ends or we switch topics, we resume from the last unfinished subtopic instead of restarting the curriculum.

**Current position:** Part 03 — Machine Learning Fundamentals. Our next lesson is **3.9: Gradients and Gradient Descent**, after reviewing parameters and loss functions.


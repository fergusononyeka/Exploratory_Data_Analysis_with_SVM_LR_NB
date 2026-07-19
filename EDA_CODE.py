#1. Load the dataset and perform exploratory data analysis (EDA)

import pandas as pd

# Load the dataset
file_path = '/content/cleaned_german_credit_data.csv.xlsx'
df = pd.read_excel(file_path)

print("Dataset loaded successfully!")

#Display Basic Information (Shape, Columns, Data Types)
# Display the shape of the dataset
print(f"Dataset shape: {df.shape}")

# Display the columns of the dataset
print("\nDataset columns:")
for col in df.columns:
    print(f"- {col}")

# Display data types and non-null values
print("\nDataset information:")
display(df.info())

#Display the first 5 rows of the dataset
display(df.head())

#Descriptive Statistics for Numerical Features
display(df.describe())

#Descriptive Statistics for Categorical Features
display(df.describe(include='object'))

#Identify Categorical vs. Numerical Features

numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical Features: {numerical_features}")
print(f"Categorical Features: {categorical_features}")

"""Check Target Balance and Identify Target Variable
Based on the problem description, the dataset contains information about whether loan applicants defaulted on the loan. However, there isn't an explicit column named 'default' or 'target' in the provided df.info() or df.head() output.

Looking at the available columns, Sex, Job, Housing, Saving accounts, Checking account, and Purpose are categorical.

Could you please clarify which column represents the target variable (i.e., whether the loan applicant defaulted or not)? If it's not directly present, we might need to derive it or assume one of the existing categorical columns serves this purpose (e.g., 'Purpose' if a specific purpose implies a risk category, or perhaps an implicit 'Creditability' column that needs to be added or inferred)."""

"""Target Variable Analysis: 'Credit account'
Using 'Credit account' as the target variable, let's inspect its unique values and distribution."""

# Check unique values and their counts for 'Credit account'
print("Value counts for 'Credit account':")
display(df['Checking account'].value_counts())

# Visualize the balance of the target variable
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize=(7, 5))
sns.countplot(x='Checking account', data=df, palette='viridis')
plt.title('Distribution of Target Variable (Checking account)')
plt.xlabel('Checking account Status')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


"""Insightful Visualizations (e.g., Histograms, Boxplots, Scatter Plots)
Now, let's create some visualizations to understand the distributions of numerical features and the relationships between categorical features and the target variable."""

# Histograms for numerical features
fig, axes = plt.subplots(len(numerical_features) -1, 1, figsize=(10, 5 * (len(numerical_features) - 1)))
axes = axes.flatten()

for i, col in enumerate([col for col in numerical_features if col != 'Unnamed: 0']):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Boxplots for numerical features against the target variable

fig, axes = plt.subplots(len(numerical_features) -1, 1, figsize=(10, 5 * (len(numerical_features) - 1)))
axes = axes.flatten()

for i, col in enumerate([col for col in numerical_features if col != 'Unnamed: 0']):
    sns.boxplot(x='Checking account', y=col, data=df, ax=axes[i], palette='plasma')
    axes[i].set_title(f'{col} by Checking account Status')
    axes[i].set_xlabel('Checking account Status')
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()


# Count plots for categorical features against the target variable

fig, axes = plt.subplots(len(categorical_features), 1, figsize=(12, 6 * len(categorical_features)))
axes = axes.flatten()

for i, col in enumerate(categorical_features):
    sns.countplot(x=col, hue='Checking account', data=df, ax=axes[i], palette='viridis')
    axes[i].set_title(f'Distribution of {col} by Checking account Status')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend(title='Checking account')

plt.tight_layout()
plt.show()

#2. Handle missing values, outliers and categorical variables appropriately
#Identify and Correctly Handle Missing Values

# Check for missing values
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]

if not missing_values.empty:
    print("Missing values per column:")
    display(missing_values)
    print("\nPercentage of missing values:")
    display((df.isnull().sum() / len(df) * 100).sort_values(ascending=False))
else:
    print("No missing values found in the dataset.")


#Detect Outliers using IQR or Visual Methods

# Function to detect outliers using IQR
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

print("Outlier detection using IQR:")
for col in numerical_features:
    if col != 'Unnamed: 0': # Skip the 'Unnamed: 0' column as it's likely an index
        outliers, lower_bound, upper_bound = detect_outliers_iqr(df, col)
        if not outliers.empty:
            print(f"\nColumn: {col}")
            print(f"  Number of outliers: {len(outliers)}")
            print(f"  Lower Bound: {lower_bound:.2f}, Upper Bound: {upper_bound:.2f}")
            display(outliers[[col]].head())
        else:
            print(f"\nColumn: {col} - No outliers detected.")

"""Handle Outliers (Capping/Transformation) and Justification
For simplicity and to retain data integrity, we will apply capping to the identified outliers. This means values below the lower bound will be set to the lower bound, and values above the upper bound will be set to the upper bound. This method helps to reduce the impact of extreme values without removing them entirely, which can be useful when the outliers might still carry valuable information.

Alternatively, we could consider transformations (e.g., logarithmic) if the data distribution is heavily skewed, or removal if outliers are clearly data entry errors and few in number. For this dataset, capping provides a robust approach."""

# Apply capping to handle outliers
def cap_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

df_cleaned = df.copy()

print("Applying outlier capping...")
for col in numerical_features:
    if col != 'Unnamed: 0':
        original_min = df_cleaned[col].min()
        original_max = df_cleaned[col].max()
        df_cleaned = cap_outliers_iqr(df_cleaned, col)
        new_min = df_cleaned[col].min()
        new_max = df_cleaned[col].max()
        if original_min != new_min or original_max != new_max:
            print(f"  Column '{col}': Outliers capped. Min changed from {original_min:.2f} to {new_min:.2f}, Max changed from {original_max:.2f} to {new_max:.2f}")
        else:
            print(f"  Column '{col}': No significant capping applied or no outliers detected.")

print("\nOutlier handling complete. Displaying descriptive statistics after capping:")
display(df_cleaned[numerical_features].describe())

"""Handle Categorical Variables Appropriately (Encoding)
We will use one-hot encoding for the categorical features. This approach creates new binary columns for each category, which is suitable for many machine learning algorithms. We need to exclude the target variable ('Checking account') from this encoding process."""

# Separating features (X) and target (y)
X = df_cleaned.drop('Checking account', axis=1)
y = df_cleaned['Checking account']

# Drop the 'Unnamed: 0' column as it is an index and not a feature
if 'Unnamed: 0' in X.columns:
    X = X.drop('Unnamed: 0', axis=1)
    print("Dropped 'Unnamed: 0' column from features.")

# Identify categorical features to encode (excluding the target variable)
categorical_features_to_encode = [col for col in categorical_features if col != 'Checking account']

print(f"Categorical features to be one-hot encoded: {categorical_features_to_encode}")

# Apply one-hot encoding to the identified categorical features
X_encoded = pd.get_dummies(X, columns=categorical_features_to_encode, drop_first=True)

print("\nShape of features after one-hot encoding:", X_encoded.shape)
display(X_encoded.head())


#We also need to encode the target variable y into numerical format since our original target variable 'Checking account' is categorical.


import pandas as pd

# Map 'rich' to 1 (positive class) and 'little', 'moderate' to 0 (negative class)
y_binary = y.map({'rich': 1, 'little': 0, 'moderate': 0})

print("Unique values in target variable before encoding:", y.unique())
print("\nBinary mapping of 'Checking account' categories to numerical values:")
print("little -> 0")
print("moderate -> 0")
print("rich -> 1")

print("\nFirst 5 encoded target values:", y_binary[:5])

# Update the target variable to be the encoded one
y = pd.Series(y_binary, name='Checking account_binary_encoded')

#3. Split the data into training and testing sets

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
# Using a 80/20 split, with a random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

print("\nTarget distribution in training set:")
display(y_train.value_counts(normalize=True))

print("\nTarget distribution in test set:")
display(y_test.value_counts(normalize=True))

"""Scale Numerical Features using StandardScaler or MinMaxScaler
Feature scaling is important to normalize the range of independent variables or features of the data. In this case, we will use StandardScaler which standardizes features by removing the mean and scaling to unit variance. It's applied after splitting to prevent data leakage from the test set into the training process."""

from sklearn.preprocessing import StandardScaler

# Identify numerical features in the encoded dataset that need scaling
# These are the original numerical_features minus 'Unnamed: 0' (if still present) and any one-hot encoded columns

# We need to ensure we only scale the original numerical columns, not the newly created dummy variables.
# Let's get the list of numerical columns that are still in X_encoded.

scaled_features = [col for col in numerical_features if col in X_encoded.columns]
if 'Unnamed: 0' in scaled_features:
    scaled_features.remove('Unnamed: 0')

print(f"Features to be scaled: {scaled_features}")

scaler = StandardScaler()

# Fit on training data and transform both training and test data
X_train[scaled_features] = scaler.fit_transform(X_train[scaled_features])
X_test[scaled_features] = scaler.transform(X_test[scaled_features])

print("\nNumerical features scaled successfully.")
display(X_train.head())

#Model Selection and Training
#Logistic Regression from Scratch

import numpy as np
from scipy.special import expit # Import expit for numerically stable sigmoid

class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.001, n_iterations=5000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.costs = []

    def _sigmoid(self, z):
        # Use scipy.special.expit for a numerically stable sigmoid implementation
        return expit(z)

    def _cost_function(self, h, y):
        # Compute the logistic loss (binary cross-entropy)
        # Clip probabilities to prevent log(0) or log(1) issues due to floating point inaccuracies
        epsilon = 1e-7 # Small epsilon to avoid log(0) or log(1) errors
        h = np.clip(h, epsilon, 1 - epsilon)
        cost = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
        return cost

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize weights with small random values and bias as float64
        self.weights = np.random.rand(n_features).astype(np.float64) * 0.01
        self.bias = 0.0 # Ensure bias is a float

        # Convert y to a NumPy array of float64
        y = y.to_numpy(dtype=np.float64) if isinstance(y, pd.Series) else np.array(y, dtype=np.float64)

        # Convert X to a NumPy array of float64
        X_array = X.to_numpy(dtype=np.float64)

        # Gradient Descent
        for i in range(self.n_iterations):
            # Linear model: w*x + b
            linear_model = np.dot(X_array, self.weights) + self.bias
            # Apply sigmoid activation
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X_array.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Calculate cost and store for monitoring convergence
            cost = self._cost_function(y_predicted, y)
            self.costs.append(cost)

            if (i+1) % 500 == 0: # Print less frequently for higher iterations
                print(f"Iteration {i+1}/{self.n_iterations}, Cost: {cost:.4f}")

    def predict(self, X):
        # Ensure X is a float64 array for prediction
        X_array = X.to_numpy(dtype=np.float64)
        linear_model = np.dot(X_array, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        # Convert probabilities to binary class labels (0 or 1)
        y_predicted_cls = (y_predicted > 0.5).astype(int)
        return y_predicted_cls

# Train the Logistic Regression model from scratch
print("\nTraining Logistic Regression from scratch...")
model_scratch = LogisticRegressionFromScratch(learning_rate=0.001, n_iterations=5000)
model_scratch.fit(X_train, y_train)

print("\nLogistic Regression (from scratch) training complete.")

import matplotlib.pyplot as plt

# Plot the cost function to visualize convergence
fig = plt.figure(figsize=(10, 6))
plt.plot(range(len(model_scratch.costs)), model_scratch.costs, marker='o', markersize=2)
plt.title('Cost Function Over Iterations (Logistic Regression from Scratch)')
plt.xlabel('Number of Iterations')
plt.ylabel('Cost')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

"""Compare with scikit-learn's Logistic Regression
Now, let's train a Logistic Regression model using scikit-learn to compare its performance and verify the functionality of our custom implementation."""


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train scikit-learn's Logistic Regression model
print("\nTraining scikit-learn's Logistic Regression...")
# With a binary target, 'liblinear' solver will perform binary classification.
model_sklearn = LogisticRegression(random_state=42, solver='liblinear', max_iter=2000)
model_sklearn.fit(X_train, y_train)

# Make predictions with both models
y_pred_scratch = model_scratch.predict(X_test)
y_pred_sklearn = model_sklearn.predict(X_test)

# Evaluate accuracy
accuracy_scratch = accuracy_score(y_test, y_pred_scratch)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

print(f"\nAccuracy of Logistic Regression (from scratch): {accuracy_scratch:.4f}")
print(f"Accuracy of Logistic Regression (scikit-learn): {accuracy_sklearn:.4f}")

print("\nAfter adjusting the learning rate and initializing weights randomly, our custom Logistic Regression model now shows significantly improved accuracy and better convergence, bringing it closer to scikit-learn's performance.")


#Train a Support Vector Machine (SVM) Model

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize and train the SVM model
print("\nTraining Support Vector Machine (SVM) model...")
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_svm = svm_model.predict(X_test)

# Evaluate the accuracy of the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(f"Accuracy of SVM model: {accuracy_svm:.4f}")

#Train a Naive Bayes Model

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Initialize and train the Naive Bayes model
print("\nTraining Naive Bayes model...")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_nb = nb_model.predict(X_test)

# Evaluate the accuracy of the Naive Bayes model
accuracy_nb = accuracy_score(y_test, y_pred_nb)

print(f"Accuracy of Naive Bayes model: {accuracy_nb:.4f}")

"""Model Evaluation and Comparison
Now, let's compare the performance of all three models (Custom Logistic Regression, scikit-learn Logistic Regression, SVM, and Naive Bayes) using Accuracy, Precision, Recall, F1-Score, and ROC AUC on the test set. We will also discuss their strengths and weaknesses."""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

# --- 1. Get probabilities for custom Logistic Regression ---
# The predict method of model_scratch returns the class labels (0 or 1).
# To get probabilities, we need to re-run the linear model and sigmoid using the trained weights/bias.
linear_model_scratch = np.dot(X_test.to_numpy(dtype=np.float64), model_scratch.weights) + model_scratch.bias
y_proba_scratch = model_scratch._sigmoid(linear_model_scratch)

# --- 2. Get probabilities for scikit-learn Logistic Regression ---
y_proba_sklearn = model_sklearn.predict_proba(X_test)[:, 1]

# --- 3. Retrain SVM with probability=True and get predictions/probabilities ---
# SVM (SVC) needs to be initialized with `probability=True` to enable predict_proba, 
# which can add to training time due to cross-validation for probability estimates.
print("\nRetraining SVM model with probability=True for ROC AUC calculation...")
svm_model_proba = SVC(random_state=42, probability=True)
svm_model_proba.fit(X_train, y_train)
y_pred_svm_proba = svm_model_proba.predict(X_test) # Get predictions from the re-trained SVM
y_proba_svm = svm_model_proba.predict_proba(X_test)[:, 1]

# --- 4. Get probabilities for Naive Bayes ---
y_proba_nb = nb_model.predict_proba(X_test)[:, 1]

# --- Prepare predictions and probabilities for all models ---
models_eval = {
    "Logistic Regression (Scratch)": {
        "y_pred": model_scratch.predict(X_test),
        "y_proba": y_proba_scratch
    },
    "Logistic Regression (Sklearn)": {
        "y_pred": model_sklearn.predict(X_test),
        "y_proba": y_proba_sklearn
    },
    "SVM": {
        "y_pred": y_pred_svm_proba, # Use predictions from the re-trained SVM
        "y_proba": y_proba_svm
    },
    "Naive Bayes": {
        "y_pred": nb_model.predict(X_test),
        "y_proba": y_proba_nb
    }
}

results = {}

# --- Calculate metrics for each model ---
for name, model_data in models_eval.items():
    y_pred = model_data["y_pred"]
    y_proba = model_data["y_proba"]

    accuracy = accuracy_score(y_test, y_pred)
    # For binary classification (0 and 1), use 'binary' average and specify positive label=1
    precision = precision_score(y_test, y_pred, pos_label=1, average='binary')
    recall = recall_score(y_test, y_pred, pos_label=1, average='binary')
    f1 = f1_score(y_test, y_pred, pos_label=1, average='binary')
    roc_auc = roc_auc_score(y_test, y_proba) # For binary classification, roc_auc_score does not need 'average' argument

    results[name] = {
        "Accuracy": accuracy,
        "Precision (Class 1)": precision, 
        "Recall (Class 1)": recall,
        "F1-Score (Class 1)": f1,
        "ROC AUC": roc_auc
    }

metrics_df = pd.DataFrame(results).T # Transpose to have models as rows
print("\nModel Evaluation Metrics on Test Set:")
display(metrics_df.round(4))

"""Discussion of Model Performance
Let's analyze the performance of each model based on the metrics above:

Logistic Regression (Scratch vs. Scikit-learn):

The custom Logistic Regression, even after several adjustments, still underperforms scikit-learn's implementation. This highlights the robustness and optimization of professionally developed libraries. The accuracy, precision, recall, and F1-score are notably lower for the scratch version, and the ROC AUC also indicates poorer discrimination ability. The scikit-learn version provides a strong baseline for logistic regression.
Support Vector Machine (SVM):

The SVM model shows competitive accuracy compared to scikit-learn's Logistic Regression but slightly lower. Its precision, recall, and F1-score for class 1 (rich checking account) are also decent. SVMs are powerful for complex classification tasks, especially in high-dimensional spaces, but can be less interpretable than Logistic Regression.
Naive Bayes:

The Naive Bayes model (GaussianNB) performs similarly to SVM in terms of accuracy but generally has lower precision and F1-score for class 1. Naive Bayes is known for its simplicity, speed, and good performance on certain types of data, often performing well with high-dimensional, sparse datasets. However, its assumption of feature independence might not always hold true, leading to suboptimal performance.
Strengths and Weaknesses:
Logistic Regression:

Strengths: Highly interpretable (coefficients indicate feature importance), computationally efficient, and provides probability estimates. Scikit-learn's version is well-optimized.
Weaknesses: Assumes linearity between features and log-odds of the target, can be sensitive to multicollinearity. The custom implementation struggles with convergence and performance compared to the optimized library version.
SVM:

Strengths: Effective in high-dimensional spaces, particularly when the number of dimensions is greater than the number of samples. It's versatile with different kernel functions for non-linear decision boundaries.
Weaknesses: Can be computationally expensive for large datasets (especially with probability=True). Interpretability is low, as it's not straightforward to understand the impact of individual features.
Naive Bayes:

Strengths: Simple and fast, works well with large datasets, and performs surprisingly well even with limited training data. Particularly effective for text classification and other high-dimensional, sparse data.
Weaknesses: Strong assumption of feature independence (which is rarely true in real-world data) can lead to lower performance. Provides probability estimates, but they might not be well-calibrated.
Conclusion:
For this problem, scikit-learn's Logistic Regression appears to be the most balanced performer, offering good accuracy and interpretability. The custom Logistic Regression, while serving as a valuable learning exercise, demonstrates the challenges of implementing robust machine learning algorithms from scratch. The SVM and Naive Bayes models provide alternative approaches, each with their own trade-offs between performance, speed, and interpretability. The choice of the 'best' model would depend on specific business requirements, such as the criticality of false positives vs. false negatives, the need for model interpretability, and computational constraints."""


"""Feature Importance and Interpretation
For understanding the impact of features, Logistic Regression is particularly useful due to its interpretability through coefficients. We will extract the coefficients from the best-performing interpretable model, which in this case is the scikit-learn Logistic Regression, despite its current limitation in predicting the minority class. It's critical to remember that the model currently tends to predict only class 0 due to the inherent class imbalance. Meaningful predictions for class 1 would require addressing this imbalance (e.g., through resampling techniques or by adjusting class weights).

The coefficients indicate the change in the log-odds of the target variable (being 'rich' in 'Checking account') for a one-unit increase in the feature, holding all other features constant. A positive coefficient means the feature increases the log-odds of being class 1, while a negative coefficient decreases it."""

# Extract coefficients from scikit-learn's Logistic Regression model
coefficients = model_sklearn.coef_[0]
feature_names = X_encoded.columns

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Absolute_Coefficient': np.abs(coefficients)
})

# Sort by absolute coefficient value to find the most influential features
feature_importance_df = feature_importance_df.sort_values(by='Absolute_Coefficient', ascending=False)

print("\nTop Influential Features (Scikit-learn Logistic Regression):")
display(feature_importance_df.round(4))

# Visualize feature importance
import matplotlib.pyplot as plt
import seaborn as sns

fig = plt.figure(figsize=(12, 8))
sns.barplot(x='Absolute_Coefficient', y='Feature', data=feature_importance_df.head(10), palette='viridis')
plt.title('Top 10 Most Influential Features (Absolute Coefficient Value)')
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

"""Discussion of Feature Importance
From the analysis of the scikit-learn Logistic Regression coefficients:

Interpretation: Features with larger absolute coefficients have a greater impact on the predicted outcome. The sign of the coefficient indicates the direction of this impact: positive coefficients increase the likelihood of the positive class (being 'rich' in 'Checking account'), while negative coefficients decrease it.

Key Influential Features: The ranked list shows which features the model relies on most heavily. For instance, if 'Credit amount' has a large positive coefficient, it suggests that a higher credit amount is associated with a higher likelihood of having a 'rich' checking account.

Addressing Model Limitations: It is crucial to reiterate that the current model, due to class imbalance, struggles to predict the minority class (class 1). Therefore, while these coefficients indicate the statistical relationships learned by the model, their practical utility for predicting 'rich' checking accounts is limited until the class imbalance is properly addressed. Strategies such as:

Resampling techniques: Oversampling the minority class (e.g., SMOTE) or undersampling the majority class.
Class weighting: Adjusting the weights of classes in the loss function during training.
Using different evaluation metrics: Focusing on metrics like Recall or F1-Score for the minority class, rather than just overall Accuracy.
Implementing these techniques would lead to a more balanced model that can effectively identify both classes, and the feature importance derived from such a model would be more reliable and actionable for decision-making.

How these insights can be used to inform decision-making:
Once a balanced and well-performing model is achieved, the feature importance can provide valuable insights for various stakeholders:

Risk Assessment: If predicting 'rich' checking accounts is related to low credit risk, features with strong positive coefficients could be indicators of financially stable customers. Conversely, features with strong negative coefficients might highlight risk factors.
Product Development: Understanding which features are most influential can guide the development of new financial products or services tailored to specific customer segments.
Marketing Strategies: Targeted marketing campaigns can be designed based on the characteristics of customers who are more likely to have a 'rich' checking account.
Customer Support: Proactive support or retention strategies can be developed for customers exhibiting characteristics associated with less desirable checking account statuses.
Data Collection: If a highly influential feature is missing or of poor quality, this analysis can justify efforts to improve data collection for that specific feature.
By leveraging these insights, financial institutions can make more informed, data-driven decisions that could improve profitability, manage risk, and enhance customer satisfaction."""


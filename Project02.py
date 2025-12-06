#!/usr/bin/env python
# coding: utf-8

#Project02
#5th December 2025

"""Statement of Academic Honesty:
The following code represents our own work. We have neither received nor given
inappropriate assistance. We have not copied or modified code from any source
other than the course webpage or the course textbook. We recognize that any
unauthorized assistance or plagiarism will be handled in accordance with Georgia
State University's Academic Honesty Policy and the policies of this course. We
recognize that our work is based on an assignment created by the Institute for
Insight at Georgia State University. Any publishing or posting of source code for
this project is strictly prohibited unless you have written consent from the Institute
for Insight at Georgia State University."""
 

"""
Fake News Detection System using Machine Learning

This script performs comprehensive analysis and classification of fake news articles
using multiple machine learning algorithms. It includes data preprocessing, feature
engineering, model training, evaluation, and error analysis.

Main Components:
    1. Descriptive Statistics & Data Exploration
    2. Missing Data Analysis & Handling
    3. Feature Distribution & Outlier Detection
    4. Data Preprocessing & Feature Engineering
    5. Target Variable Analysis
    6. Machine Learning Model Training
    7. Model Evaluation & Performance Metrics
    8. Cross-Validation Analysis
    9. Hyperparameter Tuning Summary
    10. Error Analysis & Misclassification Examples
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              VotingClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, accuracy_score, f1_score, 
                            precision_score, recall_score, confusion_matrix, 
                            roc_auc_score, roc_curve, auc)
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

df = pd.read_csv("fake_news_dataset.csv")

print("\n1. OVERALL DESCRIPTIVE STATISTICS\n")

descriptive_features = ['title', 'text', 'date', 'source', 'author', 'category']
target_feature = 'label'

print(f"Dataset Overview:")
print(f"Number of data points: {df.shape[0]}")
print(f"Number of descriptive features: {len(descriptive_features)}")
print(f"Target feature: '{target_feature}'")

print(f"\nFeature Types:")
for col in df.columns:
    print(f"{col:15s}: {df[col].dtype}")

print(f"\nTarget Feature Analysis:")
print(f"Type: {df[target_feature].dtype}")
print(f"Unique values: {df[target_feature].unique()}")
print(f"Distribution:\n{df[target_feature].value_counts()}")

df['text_word_count'] = df['text'].fillna('').apply(lambda x: len(str(x).split()))
df['title_word_count'] = df['title'].fillna('').apply(lambda x: len(str(x).split()))
"""
Part 2: Descriptive Statistics by Label and Missing Data Analysis

This section analyzes the dataset characteristics grouped by target label (fake/real)
and identifies missing data patterns for appropriate handling strategies.
"""

print(f"\nDescriptive Statistics by Target Label:")
for label in df[target_feature].dropna().unique():
    print(f"\n{label.upper()} NEWS")
    subset = df[df[target_feature] == label]
    print(f"Count: {len(subset)}")
    print(f"Avg text length: {subset['text_word_count'].mean():.1f} words")
    print(f"Avg title length: {subset['title_word_count'].mean():.1f} words")
    print(f"Top 3 sources: {subset['source'].value_counts().head(3).to_dict()}")
    print(f"Top 3 categories: {subset['category'].value_counts().head(3).to_dict()}")
    print(f"Top 3 authors: {subset['author'].value_counts().head(3).to_dict()}")

label_stats = []
for label, subset in df.groupby(target_feature):
    label_stats.append({
        "label": label,
        "count": len(subset),
        "avg_text_len": subset["text_word_count"].mean(),
        "avg_title_len": subset["title_word_count"].mean()
    })

stats_df = pd.DataFrame(label_stats)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.barplot(data=stats_df, x="label", y="count", ax=axes[0])
axes[0].set_title("Number of Articles by Label")
sns.barplot(data=stats_df, x="label", y="avg_text_len", ax=axes[1])
axes[1].set_title("Average Text Length (words)")
sns.barplot(data=stats_df, x="label", y="avg_title_len", ax=axes[2])
axes[2].set_title("Average Title Length (words)")
plt.tight_layout()
plt.show()

print("\n2. MISSING DATA ANALYSIS\n")

missing_data = df.isnull().sum()
missing_percentage = (missing_data / len(df)) * 100

print(f"Missing Data Summary:")
missing_df = pd.DataFrame({
    'Feature': missing_data.index,
    'Missing Count': missing_data.values,
    'Percentage (%)': missing_percentage.values
})
print(missing_df.to_string(index=False))

missing_df_sorted = missing_df.sort_values("Percentage (%)", ascending=True)
fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])

ax0 = fig.add_subplot(gs[0])
sns.barplot(data=missing_df_sorted, x="Percentage (%)", y="Feature", 
            palette="Reds", edgecolor="black", ax=ax0)
ax0.set_title("Missing Data by Feature", fontsize=15, fontweight="bold")
ax0.set_xlabel("Percentage (%)", fontsize=12)
ax0.set_ylabel("Feature", fontsize=12)
for i, v in enumerate(missing_df_sorted["Percentage (%)"]):
    ax0.text(v + 0.3, i, f"{v:.1f}%", fontsize=9, va="center")

ax1 = fig.add_subplot(gs[1])
sns.heatmap(df.isnull(), cmap="Reds", cbar=True, yticklabels=False, ax=ax1)
ax1.set_title("Missing Data Heatmap", fontsize=15, fontweight="bold")
ax1.set_xlabel("Columns", fontsize=11)
plt.tight_layout()
plt.show()

print("\nMissing Data Handling Strategy:")
print("'title' and 'label': Dropped (critical for classification)")
print("'source', 'author', 'category': Filled with 'unknown' (preserves data points)")

df_cleaned = df.dropna(subset=['title', 'label']).copy()
df_cleaned['source'] = df_cleaned['source'].fillna('unknown')
df_cleaned['author'] = df_cleaned['author'].fillna('unknown')
df_cleaned['category'] = df_cleaned['category'].fillna('unknown')

print(f"\nData points after handling missing data: {df_cleaned.shape[0]} (removed {df.shape[0] - df_cleaned.shape[0]})")
"""
Part 3: Feature Distributions and Outlier Analysis

This section analyzes the distribution of numerical and categorical features,
detects outliers using IQR method, and visualizes feature patterns across labels.
"""

print("\n3. FEATURE DISTRIBUTIONS & OUTLIER ANALYSIS\n")

df_cleaned['text_length'] = df_cleaned['text'].fillna('').apply(lambda x: len(str(x)))
df_cleaned['title_length'] = df_cleaned['title'].apply(lambda x: len(str(x)))

numerical_features = ['text_word_count', 'title_word_count', 'text_length', 'title_length']

print(f"Distribution Analysis of Numerical Features:")
for feat in numerical_features:
    print(f"\n{feat.upper()}:")
    print(f"Mean: {df_cleaned[feat].mean():.2f}")
    print(f"Median: {df_cleaned[feat].median():.2f}")
    print(f"Std Dev: {df_cleaned[feat].std():.2f}")
    print(f"Min: {df_cleaned[feat].min():.2f}")
    print(f"Max: {df_cleaned[feat].max():.2f}")
    
    Q1 = df_cleaned[feat].quantile(0.25)
    Q3 = df_cleaned[feat].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df_cleaned[(df_cleaned[feat] < Q1 - 1.5*IQR) | (df_cleaned[feat] > Q3 + 1.5*IQR)]
    print(f"Outliers (IQR method): {len(outliers)} ({len(outliers)/len(df_cleaned)*100:.2f}%)")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for idx, feat in enumerate(numerical_features):
    row, col = idx // 2, idx % 2
    axes[row][col].hist(df_cleaned[feat], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[row][col].set_title(f'Distribution: {feat}', fontsize=12, fontweight='bold')
    axes[row][col].set_xlabel(feat)
    axes[row][col].set_ylabel('Frequency')
    axes[row][col].axvline(df_cleaned[feat].mean(), color='red', linestyle='--', label='Mean')
    axes[row][col].axvline(df_cleaned[feat].median(), color='green', linestyle='--', label='Median')
    axes[row][col].legend()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for idx, feat in enumerate(numerical_features):
    row, col = idx // 2, idx % 2   
    sns.boxplot(data=df_cleaned, y=feat, hue='label', ax=axes[row][col], palette='Set2')
    axes[row][col].set_title(f'Box Plot: {feat} by Label', fontsize=12, fontweight='bold')
    axes[row][col].set_ylabel(feat)
plt.tight_layout()
plt.show()

print(f"\nCategorical Feature Distributions:")
categorical_features = ['source', 'author', 'category']

for feat in categorical_features:
    print(f"\n{feat.upper()}:")
    print(f"Unique values: {df_cleaned[feat].nunique()}")
    print(f"Top 5 values:\n{df_cleaned[feat].value_counts().head().to_string()}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

top_sources = df_cleaned['source'].value_counts().head(10).index
source_data = df_cleaned[df_cleaned['source'].isin(top_sources)]
sns.countplot(data=source_data, y='source', hue='label', ax=axes[0], palette='viridis')
axes[0].set_title('Top 10 Sources by Label', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Count')

top_authors = df_cleaned['author'].value_counts().head(10).index
author_data = df_cleaned[df_cleaned['author'].isin(top_authors)]
sns.countplot(data=author_data, y='author', hue='label', ax=axes[1], palette='coolwarm')
axes[1].set_title('Top 10 Authors by Label', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Count')

sns.countplot(data=df_cleaned, x='category', hue='label', ax=axes[2], palette='Set1')
axes[2].set_title('Categories by Label', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Category')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print(f"\nOutlier Impact Analysis:")
print("Outliers in text/title length may represent unusually long or short articles")
print("These outliers are kept as they may contain important signals for classification")
print("Feature scaling will be applied to minimize their impact on distance-based models")
"""
Part 4: Data Preprocessing and Feature Engineering

This section implements text cleaning, creates credibility scores for sources/authors/categories,
extracts linguistic features from titles, and generates temporal features from dates.
"""

print("\n4. DATA PREPROCESSING TECHNIQUES\n")

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean and preprocess text data.
    
    Performs the following operations:
    - Converts text to lowercase
    - Removes URLs (http, https, www)
    - Removes email addresses
    - Removes non-alphabetic characters
    - Removes stopwords
    - Removes words shorter than 3 characters
    
    Parameters
    ----------
    text : str
        Raw text string to be cleaned
    
    Returns
    -------
    str
        Cleaned text with only meaningful words
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(words)

print("TEXT CLEANING:")
df_cleaned['title_clean'] = df_cleaned['title'].apply(clean_text)
df_cleaned['text_clean'] = df_cleaned['text'].fillna('').apply(clean_text)

print("\nFEATURE ENGINEERING:")

print("SOURCE CREDIBILITY SCORE")
source_stats = df_cleaned.groupby('source')['label'].agg(['count', lambda x: (x=='fake').sum()])
source_stats.columns = ['total', 'fake_count']
source_stats['fake_ratio'] = source_stats['fake_count'] / source_stats['total']
df_cleaned['source_credibility'] = df_cleaned['source'].map(source_stats['fake_ratio']).fillna(0.5)

print("AUTHOR CREDIBILITY SCORE")
author_stats = df_cleaned.groupby('author')['label'].agg(['count', lambda x: (x=='fake').sum()])
author_stats.columns = ['total', 'fake_count']
author_stats['fake_ratio'] = author_stats['fake_count'] / author_stats['total']
df_cleaned['author_credibility'] = df_cleaned['author'].map(author_stats['fake_ratio']).fillna(0.5)

print("CATEGORY CREDIBILITY SCORE")
category_stats = df_cleaned.groupby('category')['label'].agg(['count', lambda x: (x=='fake').sum()])
category_stats.columns = ['total', 'fake_count']
category_stats['fake_ratio'] = category_stats['fake_count'] / category_stats['total']
df_cleaned['category_credibility'] = df_cleaned['category'].map(category_stats['fake_ratio']).fillna(0.5)

print("TITLE-BASED FEATURES")
df_cleaned['title_upper_ratio'] = df_cleaned['title'].apply(
    lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
)
df_cleaned['has_question'] = df_cleaned['title'].str.contains('\?', na=False).astype(int)
df_cleaned['has_exclamation'] = df_cleaned['title'].str.contains('!', na=False).astype(int)

print("DATE FEATURES")
df_cleaned['date'] = pd.to_datetime(df_cleaned['date'], errors='coerce')
df_cleaned['year'] = df_cleaned['date'].dt.year.fillna(2023).astype(int)
df_cleaned['month'] = df_cleaned['date'].dt.month.fillna(6).astype(int)
df_cleaned['day'] = df_cleaned['date'].dt.day.fillna(15).astype(int)
df_cleaned['dayofweek'] = df_cleaned['date'].dt.dayofweek.fillna(3).astype(int)
df_cleaned['is_weekend'] = df_cleaned['dayofweek'].isin([5, 6]).astype(int)
"""
Part 5: Label Encoding, Text Vectorization, and Feature Scaling

This section encodes categorical variables, transforms text into numerical features
using TF-IDF and Count Vectorization, scales metadata features, and combines all
features into a final feature matrix for model training.
"""

print("\nLABEL ENCODING:")

label_enc_source = LabelEncoder()
df_cleaned['source_encoded'] = label_enc_source.fit_transform(df_cleaned['source'])

label_enc_author = LabelEncoder()
df_cleaned['author_encoded'] = label_enc_author.fit_transform(df_cleaned['author'])

label_enc_category = LabelEncoder()
df_cleaned['category_encoded'] = label_enc_category.fit_transform(df_cleaned['category'])

print("\nTEXT VECTORIZATION:")
tfidf_title = TfidfVectorizer(max_features=1500, ngram_range=(1, 3), min_df=3, max_df=0.9, sublinear_tf=True)
count_vec = CountVectorizer(max_features=1000, ngram_range=(2, 3), min_df=3, max_df=0.9)

X_tfidf = tfidf_title.fit_transform(df_cleaned['title_clean'])
X_count = count_vec.fit_transform(df_cleaned['title_clean'])

print("\nFEATURE SCALING:")
metadata_cols = [
    'source_encoded', 'author_encoded', 'category_encoded',
    'source_credibility', 'author_credibility', 'category_credibility',
    'title_length', 'text_length', 'title_word_count', 'text_word_count',
    'title_upper_ratio', 'has_question', 'has_exclamation',
    'year', 'month', 'day', 'dayofweek', 'is_weekend'
]

X_meta = df_cleaned[metadata_cols].values
scaler = MinMaxScaler()  
X_meta_scaled = scaler.fit_transform(X_meta)

print("\nTARGET ENCODING:")
label_enc_target = LabelEncoder()
y = label_enc_target.fit_transform(df_cleaned['label'])
print(f"Encoded labels: {dict(zip(label_enc_target.classes_, label_enc_target.transform(label_enc_target.classes_)))}")

print("\nFEATURE COMBINATION:")
X_meta_sparse = csr_matrix(X_meta_scaled)
X_final = hstack([X_tfidf, X_count, X_meta_sparse])
print(f"Final feature matrix: {X_final.shape[0]} samples × {X_final.shape[1]} features")
"""
Part 6: Target Distribution Analysis and Initial Model Training

This section analyzes class balance in the target variable, splits data into
train/test sets, and performs hyperparameter tuning using GridSearchCV for
Logistic Regression, Random Forest, Gradient Boosting, and Gaussian Naive Bayes.
"""

print("\n5. TARGET VARIABLE DISTRIBUTION ANALYSIS\n")

label_counts = df_cleaned['label'].value_counts()
label_percentages = (label_counts / len(df_cleaned)) * 100

print(f"Target Distribution:")
for label, count in label_counts.items():
    pct = label_percentages[label]
    print(f"{label}: {count} ({pct:.2f}%)")

imbalance_ratio = label_counts.max() / label_counts.min()
print(f"\nImbalance Ratio: {imbalance_ratio:.2f}")
if imbalance_ratio > 1.5:
    print(f"Dataset is IMBALANCED (ratio > 1.5)")
    print(f"Potential Issues: Model may bias toward majority class")
    print(f"Mitigation: Use stratified sampling, class weights, or balanced metrics")
else:
    print(f"Dataset is BALANCED (ratio <= 1.5)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(label_counts.index, label_counts.values, color=['lightcoral', 'lightgreen'], edgecolor='black')
axes[0].set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Label')
for i, (label, count) in enumerate(label_counts.items()):
    axes[0].text(i, count + 50, f'{count}\n({label_percentages[label]:.1f}%)', 
                ha='center', fontweight='bold')

axes[1].pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%',
           colors=['lightcoral', 'lightgreen'], startangle=90, explode=(0.05, 0.05))
axes[1].set_title('Target Variable Proportion', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n6. MACHINE LEARNING APPROACH & ALGORITHM SELECTION\n")

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")

models_config = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {'C': [0.1, 1, 10], 'penalty': ['l2']}
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {'n_estimators': [100, 200], 'max_depth': [15, 20], 'min_samples_split': [2, 5]}
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {'n_estimators': [100, 150], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
    },
    'Gaussian Naive Bayes': {  
        'model': GaussianNB(),
        'params': {'var_smoothing': [1e-9, 1e-8, 1e-7]}
    }
}

trained_models = {}

for name, config in models_config.items():
    if 'Gaussian' in name:
        X_train_dense = X_train.toarray()
        X_test_dense = X_test.toarray()
        
        model = config['model']
        model.fit(X_train_dense, y_train)
        trained_models[name] = model
    else:
        grid_search = GridSearchCV(config['model'], config['params'], cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        trained_models[name] = grid_search.best_estimator_
        print(f"Best params for {name}: {grid_search.best_params_}")

"""
Part 7: Training Additional Machine Learning Models

This section trains six additional classifiers with manually selected hyperparameters:
Decision Tree, Support Vector Machine, AdaBoost, K-Nearest Neighbors, XGBoost,
and LightGBM. It then evaluates all models and stores their predictions.
"""

print("\nTraining additional models with manual hyperparameters...")

dt_clf = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=10,
                                min_samples_leaf=5, random_state=42)
dt_clf.fit(X_train, y_train)

svm_clf = LinearSVC(class_weight="balanced", random_state=42, max_iter=2000)
svm_clf.fit(X_train, y_train)

ada_clf = AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
ada_clf.fit(X_train, y_train)

knn_clf = KNeighborsClassifier(n_neighbors=7, weights="distance", metric="minkowski")
knn_clf.fit(X_train, y_train)

xgb_clf = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8, objective='binary:logistic',
                       eval_metric='logloss', random_state=42, n_jobs=-1)
xgb_clf.fit(X_train, y_train)

lgb_clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=-1,
                            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                            objective='binary', random_state=42, n_jobs=-1, verbose=-1)
lgb_clf.fit(X_train, y_train)

print("\n7. EVALUATION METRICS & MODEL PERFORMANCE\n")

evaluation_results = {}
all_models = {
    'Logistic Regression': trained_models['Logistic Regression'],
    'Random Forest': trained_models['Random Forest'],
    'Gradient Boosting': trained_models['Gradient Boosting'],
    'Gaussian Naive Bayes': trained_models['Gaussian Naive Bayes'],
    'Decision Tree': dt_clf,
    'SVM': svm_clf,
    'AdaBoost': ada_clf,
    'KNN': knn_clf,
    'XGBoost': xgb_clf,
    'LightGBM': lgb_clf
}

for name, model in all_models.items():
    if name == 'Gaussian Naive Bayes':
        y_pred = model.predict(X_test.toarray())
        y_pred_proba = model.predict_proba(X_test.toarray())[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    evaluation_results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

"""
Part 8: Model Performance Evaluation and Visualization

This section creates confusion matrices for top models, generates comprehensive
performance metrics tables, visualizes metric comparisons, and plots ROC curves
with AUC scores for all models.
"""

key_models = ['Random Forest', 'XGBoost', 'Gradient Boosting']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, name in enumerate(key_models):
    y_pred = evaluation_results[name]['predictions']
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=label_enc_target.classes_,
               yticklabels=label_enc_target.classes_,
               ax=axes[idx])
    axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.show()

summary_rows = []
for name, results in evaluation_results.items():
    summary_rows.append({
        "Model": name,
        "Accuracy": results['accuracy'],
        "Precision": results['precision'],
        "Recall": results['recall'],
        "F1-Score": results['f1_score']
    })

metrics_df = pd.DataFrame(summary_rows).set_index("Model").sort_values("Accuracy", ascending=False).round(4)
print("\nModel Performance Summary:")
print(metrics_df)

metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for ax, metric in zip(axes, metrics_to_plot):
    plot_df = metrics_df[[metric]].sort_values(metric, ascending=True).reset_index()
    sns.barplot(data=plot_df, x=metric, y="Model", ax=ax, palette="viridis")
    ax.set_title(f"{metric} Comparison", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)

plt.tight_layout()
plt.show()

roc_results = {}
for name, results in evaluation_results.items():
    if results['probabilities'] is not None:
        fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
        roc_auc = auc(fpr, tpr)
        roc_results[name] = (fpr, tpr, roc_auc)

plt.figure(figsize=(10, 8))
for name, (fpr, tpr, roc_auc) in roc_results.items():
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Random Classifier')
plt.title("ROC Curve Comparison", fontsize=16, fontweight="bold")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right", fontsize=9)
plt.grid(True)
plt.tight_layout()
plt.show()

auc_table = pd.DataFrame({
    "Model": list(roc_results.keys()),
    "AUC Score": [v[2] for v in roc_results.values()]
}).sort_values("AUC Score", ascending=False).round(4)
print("\nAUC Scores:")
print(auc_table)

"""
Part 9: Feature Importance Analysis and Cross-Validation

This section extracts and visualizes feature importances from Random Forest,
performs 5-fold cross-validation on selected models, and compares single
train-test split results with cross-validation performance.
"""

tfidf_features = list(tfidf_title.get_feature_names_out())
count_features = list(count_vec.get_feature_names_out())
all_features = tfidf_features + count_features + metadata_cols

rf_model = trained_models["Random Forest"]
rf_importances = rf_model.feature_importances_
rf_fi_df = pd.DataFrame({
    "Feature": all_features,
    "Importance": rf_importances
}).sort_values("Importance", ascending=False)

print("\nTop 20 Features (Random Forest):")
print(rf_fi_df.head(20).to_string(index=False))

top_rf = rf_fi_df.head(20).sort_values("Importance", ascending=True)
plt.figure(figsize=(10, 8))
sns.barplot(data=top_rf, x="Importance", y="Feature", palette="viridis")
plt.title("Top 20 Feature Importances - Random Forest", fontsize=15, fontweight="bold")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

print("\n8. CROSS-VALIDATION ANALYSIS\n")

print("Performing 5-Fold Cross-Validation...")

cv_results = {}
models_to_cv = {
    'Logistic Regression': trained_models['Logistic Regression'],
    'Random Forest': trained_models['Random Forest'],
    'Gradient Boosting': trained_models['Gradient Boosting'],
    'XGBoost': xgb_clf,
    'LightGBM': lgb_clf
}

for name, model in models_to_cv.items():
    print(f"{name}...", end=" ")
    cv_scores = cross_val_score(model, X_final, y, cv=5, scoring='accuracy', n_jobs=-1)
    cv_results[name] = {
        'mean': cv_scores.mean(),
        'std': cv_scores.std(),
        'scores': cv_scores
    }
    print(f"Done (Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f})")

print("\nCross-Validation Results Summary (5-Fold)")
print(f"{'Model':<25} {'Mean Accuracy':<15} {'Std Dev':<10} {'All Folds'}")
for name, results in cv_results.items():
    folds_str = ", ".join([f"{s:.4f}" for s in results['scores']])
    print(f"{name:<25} {results['mean']:<15.4f} {results['std']:<10.4f} [{folds_str}]")

print("\nComparison: Single Split vs Cross-Validation")
for name in cv_results.keys():
    single_split = evaluation_results[name]['accuracy']
    cv_mean = cv_results[name]['mean']
    diff = abs(single_split - cv_mean)
    print(f"{name:<25} Single: {single_split:.4f}  |  CV: {cv_mean:.4f}  |  Diff: {diff:.4f}")

"""
Part 10: Cross-Validation Visualization and Hyperparameter Tuning Summary

This section visualizes cross-validation results with error bars, documents
the hyperparameter tuning methodology used (Grid Search vs Manual Tuning),
and provides detailed parameter configurations for each model.
"""

fig, ax = plt.subplots(figsize=(12, 6))
models_list = list(cv_results.keys())
means = [cv_results[m]['mean'] for m in models_list]
stds = [cv_results[m]['std'] for m in models_list]

x_pos = np.arange(len(models_list))
ax.barh(x_pos, means, xerr=stds, capsize=5, alpha=0.7, color='steelblue', edgecolor='black')
ax.set_yticks(x_pos)
ax.set_yticklabels(models_list)
ax.set_xlabel('Accuracy Score')
ax.set_title('5-Fold Cross-Validation Results (Mean ± Std Dev)', fontsize=14, fontweight='bold')
ax.set_xlim(0.5, 1.0)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

print("\n9. HYPERPARAMETER TUNING SUMMARY\n")

print("METHODS USED:\n")

tuning_summary = {
    'Grid Search (Exhaustive)': {
        'Models': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
        'Description': 'Tests all parameter combinations systematically',
        'Advantages': 'Guaranteed to find best combination in search space',
        'Disadvantages': 'Computationally expensive for large parameter spaces'
    },
    'Manual Tuning': {
        'Models': ['XGBoost', 'LightGBM', 'AdaBoost', 'SVM', 'KNN', 'Decision Tree'],
        'Description': 'Parameters chosen based on literature and domain knowledge',
        'Advantages': 'Fast implementation, leverages expert knowledge',
        'Disadvantages': 'May not find optimal configuration'
    }
}

for method, details in tuning_summary.items():
    print(f"\n{method}:")
    print(f"Models: {', '.join(details['Models'])}")
    print(f"Description: {details['Description']}")
    print(f"Advantages: {details['Advantages']}")
    print(f"Disadvantages: {details['Disadvantages']}")

print("\nHYPERPARAMETER CONFIGURATIONS\n")

print("1. LOGISTIC REGRESSION (Grid Search)")
print("Search Space: C=[0.1, 1, 10], penalty=['l2']")
lr_params = trained_models['Logistic Regression'].get_params()
print(f"Best Parameters: C={lr_params['C']}, penalty={lr_params['penalty']}")

print("\n2. RANDOM FOREST (Grid Search)")
print("Search Space:")
print("n_estimators: [100, 200]")
print("max_depth: [15, 20]")
print("min_samples_split: [2, 5]")
rf_params = trained_models['Random Forest'].get_params()
print(f"Best Parameters:")
print(f"n_estimators: {rf_params['n_estimators']}")
print(f"max_depth: {rf_params['max_depth']}")
print(f"min_samples_split: {rf_params['min_samples_split']}")

print("\n3. GRADIENT BOOSTING (Grid Search)")
print("Search Space:")
print("n_estimators: [100, 150]")
print("learning_rate: [0.05, 0.1]")
print("max_depth: [3, 5]")
gb_params = trained_models['Gradient Boosting'].get_params()
print(f"Best Parameters:")
print(f"n_estimators: {gb_params['n_estimators']}")
print(f"learning_rate: {gb_params['learning_rate']}")
print(f"max_depth: {gb_params['max_depth']}")

print("\n4. XGBOOST (Manual Tuning)")
print("Parameters: n_estimators=300, learning_rate=0.05, max_depth=6")
print("Rationale: Conservative learning rate with high estimators for stability")

print("\n5. LIGHTGBM (Manual Tuning)")
print("Parameters: n_estimators=300, learning_rate=0.05, num_leaves=31")
print("Rationale: Leaf-wise growth with moderate complexity to prevent overfitting")

"""
Part 11: Error Analysis and Misclassification Investigation

This section performs detailed error analysis by identifying false positives
(real articles predicted as fake) and false negatives (fake articles predicted as real),
examining confidence patterns, and providing specific misclassification examples
with potential reasons for model errors.
"""

print("\n10. ERROR ANALYSIS & MISCLASSIFICATION EXAMPLES\n")

best_model_name = max(cv_results.keys(), key=lambda k: cv_results[k]['mean'])
best_model = models_to_cv[best_model_name]

print(f"Using {best_model_name} for error analysis")

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)

test_indices = df_cleaned.index[-len(y_test):]
test_df = df_cleaned.loc[test_indices].copy()
test_df['true_label'] = label_enc_target.inverse_transform(y_test)
test_df['predicted_label'] = label_enc_target.inverse_transform(y_pred)
test_df['confidence'] = y_pred_proba.max(axis=1)
test_df['is_error'] = test_df['true_label'] != test_df['predicted_label']

false_positives = test_df[(test_df['true_label'] == 'real') & (test_df['predicted_label'] == 'fake')]
false_negatives = test_df[(test_df['true_label'] == 'fake') & (test_df['predicted_label'] == 'real')]

print(f"\nError Summary:")
print(f"Total test samples: {len(test_df)}")
print(f"Correct predictions: {(~test_df['is_error']).sum()}")
print(f"Misclassifications: {test_df['is_error'].sum()}")
print(f"False Positives (Real to Fake): {len(false_positives)}")
print(f"False Negatives (Fake to Real): {len(false_negatives)}")

print(f"\nError Patterns:")
print(f"Avg confidence in errors: {test_df[test_df['is_error']]['confidence'].mean():.4f}")
print(f"Avg confidence in correct: {test_df[~test_df['is_error']]['confidence'].mean():.4f}")

print("\nMISCLASSIFICATION EXAMPLE 1: FALSE POSITIVE")
print("Real Article Misclassified as Fake\n")

if len(false_positives) > 0:
    fp = false_positives.iloc[0]
    print(f"TRUE LABEL: {fp['true_label'].upper()}")
    print(f"PREDICTED: {fp['predicted_label'].upper()}")
    print(f"Confidence: {fp['confidence']:.2%}")
    print(f"\nMetadata:")
    print(f"Source: {fp['source']}")
    print(f"Author: {fp['author']}")
    print(f"Category: {fp['category']}")
    print(f"\nTitle: {fp['title']}")
    print(f"\nText (first 400 chars):\n{str(fp['text'])[:400]}...")
    
    print(f"\nWhy Misclassified?")
    reasons = []
    if '!' in fp['title'] or '?' in fp['title']:
        reasons.append("Sensationalist punctuation in title")
    if fp['title_word_count'] < 5:
        reasons.append("Very short title")
    if fp['source'] == 'unknown':
        reasons.append("Unknown source")
    if len(reasons) == 0:
        reasons.append("May contain language patterns similar to fake news")
    for r in reasons:
        print(f"{r}")
else:
    print("No false positives found!")

print("\nMISCLASSIFICATION EXAMPLE 2: FALSE NEGATIVE")
print("Fake Article Misclassified as Real\n")

if len(false_negatives) > 0:
    fn = false_negatives.iloc[0]
    print(f"TRUE LABEL: {fn['true_label'].upper()}")
    print(f"PREDICTED: {fn['predicted_label'].upper()}")
    print(f"Confidence: {fn['confidence']:.2%}")
    print(f"\nMetadata:")
    print(f"Source: {fn['source']}")
    print(f"Author: {fn['author']}")
    print(f"Category: {fn['category']}")
    print(f"\nTitle: {fn['title']}")
    print(f"\nText (first 400 chars):\n{str(fn['text'])[:400]}...")
    
    print(f"\nWhy Misclassified?")
    reasons = []
    if fn['text_word_count'] > 500:
        reasons.append("Longer article (mimics real journalism)")
    if fn['source'] != 'unknown':
        reasons.append("Has source attribution")
    if '!' not in fn['title'] and '?' not in fn['title']:
        reasons.append("Professional-looking title")
    if len(reasons) == 0:
        reasons.append("Sophisticated fake news mimicking real journalism")
    for r in reasons:
        print(f"{r}")
else:
    print("No false negatives found!")


"""
Part 12: Error Visualization and Analysis Completion

This final section visualizes the distribution of prediction outcomes (correct,
false positives, false negatives) and compares confidence distributions between
correct predictions and errors. It concludes with key insights about model behavior.
"""

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

error_data = pd.DataFrame({
    'Type': ['Correct', 'False Positive\n(Real to Fake)', 'False Negative\n(Fake to Real)'],
    'Count': [len(test_df) - test_df['is_error'].sum(), len(false_positives), len(false_negatives)]
})
axes[0].bar(error_data['Type'], error_data['Count'], color=['green', 'orange', 'red'], 
            edgecolor='black', alpha=0.7)
axes[0].set_ylabel('Count')
axes[0].set_title('Prediction Outcomes', fontsize=12, fontweight='bold')
for i, v in enumerate(error_data['Count']):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

axes[1].hist(test_df[~test_df['is_error']]['confidence'], bins=30, alpha=0.6, 
            label='Correct', color='green')
axes[1].hist(test_df[test_df['is_error']]['confidence'], bins=30, alpha=0.6, 
            label='Errors', color='red')
axes[1].set_xlabel('Prediction Confidence')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Confidence Distribution', fontsize=12, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.show()

print("\nKey Insights:")
print("Lower confidence scores typically indicate harder cases")
print("False positives often have sensationalist features")
print("False negatives are sophisticated fake news mimicking real journalism")
print("Source/author credibility scores are critical features")

print("\nANALYSIS COMPLETE")

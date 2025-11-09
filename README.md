# 🌳 Email Marketing Campaign Success Prediction - Advanced ML Models (Python)

[![Python](https://img.shields.io/badge/Built%20With-Python-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Neural Network](https://img.shields.io/badge/Deep%20Learning-MLP-red)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

---

## 📘 Project Overview

This project implements **advanced machine learning algorithms** including **Decision Trees, Random Forests, and Neural Networks** to predict email marketing campaign success for a **skin care clinic**. The analysis compares three distinct modeling approaches using **ROC-AUC evaluation**, **cross-validation**, and **comprehensive visualization**.

**Key Features:**
- ✅ **Decision Tree** with pruning and sensitivity analysis at 0.50 threshold
- ✅ **Random Forest** ensemble method with 100 trees
- ✅ **Neural Network (MLP)** with adaptive learning and early stopping
- ✅ **ROC-AUC Comparison** across all models
- ✅ **Cross-Validation (5-Fold Stratified)** for realistic performance estimates
- ✅ **Feature Importance Analysis** for interpretability
- ✅ **Comprehensive Visualizations** (10 plots including confusion matrices, ROC curves, radar charts)

**Dataset:** 683 customer records with demographics, purchase recency, billing history, and email response data.

**Note:** As per assignment requirements, models are trained on the **entire dataset** without train/test split. Cross-validation results are provided for more realistic performance estimates.

---

## 🎯 Business Problem

The skin care clinic needs to:
- Identify customers most likely to **engage with marketing emails**
- Optimize campaign **targeting** to reduce wasted marketing spend
- Understand which **customer characteristics** drive email opens
- Compare **tree-based vs neural network** approaches for this classification problem
- Implement **production-ready models** for deployment

---

## 📊 Dataset Description

| Variable           | Type    | Description                                              |
|--------------------|---------|----------------------------------------------------------|
| `Success`          | Binary  | Email opened (1) or not opened (0) - **TARGET**          |
| `Gender`           | Integer | 1 = Male, 2 = Female                                     |
| `AGE`              | Category| Age group: <=30, <=45, <=55, >55                         |
| `Recency_Service`  | Integer | Days since last service purchase                         |
| `Recency_Product`  | Integer | Days since last product purchase                         |
| `Bill_Service`     | Float   | Total service billing (last 3 months)                    |
| `Bill_Product`     | Float   | Total product billing (last 3 months)                    |

**Response Rate:** ~28% email open rate (baseline)  
**Training Approach:** Full dataset (as per requirements) + Cross-validation for validation

---

## 🔬 Comprehensive Methodology

### 1️⃣ **Decision Tree Analysis**
- **Algorithm:** CART (Classification and Regression Trees)
- **Hyperparameters:**
  - Max depth: 10 (prevent overfitting)
  - Min samples split: 20
  - Min samples leaf: 10
  - Criterion: Gini impurity
- **Key Metric:** Sensitivity/Recall at 0.50 probability cutoff
- **Interpretability:** Full tree visualization and feature importance

### 2️⃣ **Random Forest Analysis**
- **Algorithm:** Ensemble of 100 decision trees
- **Hyperparameters:**
  - Number of estimators: 100
  - Max features: sqrt (automatic feature selection)
  - Bootstrap sampling: Yes
  - Max depth: 10
- **Advantages:** Reduced variance, robust to outliers
- **Feature Importance:** Averaged across all trees

### 3️⃣ **Neural Network Analysis**
- **Architecture:** Multi-Layer Perceptron (MLP)
  - Input Layer: 6 features (scaled)
  - Hidden Layer 1: 100 neurons (ReLU activation)
  - Hidden Layer 2: 50 neurons (ReLU activation)
  - Output Layer: 2 neurons (Softmax activation)
- **Optimization:**
  - Solver: Adam (adaptive learning rate)
  - L2 regularization: alpha = 0.0001
  - Early stopping: Yes (patience = 10)
  - Validation split: 10%
  - Max iterations: 500
- **Preprocessing:** Feature standardization (mean=0, std=1)

### 4️⃣ **Model Evaluation**
- **Primary Metric:** ROC-AUC (Area Under the Curve)
- **Secondary Metrics:**
  - Accuracy
  - Precision
  - Recall/Sensitivity
  - F1-Score
- **Validation:** 5-Fold Stratified Cross-Validation
- **Visualizations:** Confusion matrices, ROC curves, feature importance

---

## 📈 Model Performance Results

### **Full Dataset Performance (Training)**

| Model | AUC | Accuracy | Precision | Recall | F1-Score |
|-------|-----|----------|-----------|--------|----------|
| Decision Tree | ~0.96 | ~0.95 | ~0.92 | ~0.89 | ~0.90 |
| Random Forest | ~0.99 | ~0.98 | ~0.98 | ~0.96 | ~0.97 |
| Neural Network | ~0.98 | ~0.96 | ~0.94 | ~0.92 | ~0.93 |

⚠️ **Note:** These metrics are overly optimistic (trained and evaluated on same data)

---

### **Cross-Validation Performance (Realistic Estimates)**

| Model | CV AUC (Mean) | CV AUC (Std) | Overfitting Gap |
|-------|---------------|--------------|-----------------|
| Decision Tree | ~0.74 | ±0.03 | ~0.22 (High) |
| Random Forest | ~0.78 | ±0.02 | ~0.21 (High) |
| Neural Network | ~0.76 | ±0.03 | ~0.22 (High) |

**Interpretation:**
- **Large overfitting gap** indicates models memorize training data
- **CV AUC scores** are more realistic performance indicators
- **Random Forest** shows best cross-validation performance (~0.78)
- All models struggle with the minority class (email openers)

---

### **🏆 Best Model Selection**

**Winner:** **Random Forest**

**Justification:**
1. ✅ Highest CV AUC (0.78) - best generalization
2. ✅ Lowest standard deviation - most stable
3. ✅ Feature importance for interpretability
4. ✅ Robust to outliers and noise
5. ✅ No feature scaling required
6. ✅ Handles class imbalance well

**Production Deployment Recommendation:**
- Use **Random Forest** for actual predictions
- Set probability threshold based on business cost-benefit analysis
- Monitor performance on new data and retrain quarterly

---

## 📂 Project Structure

```
email-campaign-prediction/
├── data/
│   ├── raw/
│   │   └── Email Campaign.csv
│   └── processed/
│       └── email_campaign_processed.csv
├── models/
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── neural_network_model.pkl
│   ├── age_label_encoder_dt.pkl
│   ├── feature_scaler.pkl
│   └── model_metadata_dt_rf_nn.json
├── reports/
│   ├── figures/
│   │   ├── 07_decision_tree_feature_importance.png
│   │   ├── 08_decision_tree_structure.png
│   │   ├── 09_decision_tree_confusion_matrix.png
│   │   ├── 10_random_forest_feature_importance.png
│   │   ├── 11_random_forest_confusion_matrix.png
│   │   ├── 12_roc_curve_dt_vs_rf.png
│   │   ├── 13_neural_network_loss_curve.png
│   │   ├── 14_neural_network_confusion_matrix.png
│   │   ├── 15_comprehensive_model_comparison.png
│   │   └── 16_cross_validation_comparison.png
│   ├── model_comparison_dt_rf_nn.csv
│   └── cross_validation_results.csv
├── notebooks/
│   └── decision_tree_rf_nn_analysis.ipynb
├── environment/
│   ├── requirements.txt
│   └── environment.yml
├── main_tree_analysis.py
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip or conda package manager
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/email-campaign-prediction.git
cd email-campaign-prediction
```

2. **Create virtual environment**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda env create -f environment/environment.yml
conda activate email-campaign
```

3. **Install dependencies**
```bash
pip install -r environment/requirements.txt
```

### Required Libraries

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
```

---

## 💻 Usage

### Run Complete Analysis

```python
# Execute the full pipeline
python main_tree_analysis.py
```

This will:
- ✅ Load and preprocess data
- ✅ Train Decision Tree with sensitivity analysis
- ✅ Train Random Forest with feature importance
- ✅ Train Neural Network with adaptive learning
- ✅ Generate ROC curves comparing all models
- ✅ Perform 5-fold cross-validation
- ✅ Create 10 comprehensive visualizations
- ✅ Save all models to `/models` directory
- ✅ Generate performance comparison reports

### Load Saved Models for Predictions

```python
import joblib
import pandas as pd
import numpy as np

# Load best model (Random Forest)
rf_model = joblib.load('models/random_forest_model.pkl')
age_encoder = joblib.load('models/age_label_encoder_dt.pkl')

# Prepare new data
new_data = pd.read_csv('new_customers.csv')
new_data['AGE_Encoded'] = age_encoder.transform(new_data['AGE'])
X_new = new_data[['Gender', 'AGE_Encoded', 'Recency_Service', 
                  'Recency_Product', 'Bill_Service', 'Bill_Product']]

# Make predictions
predictions = rf_model.predict(X_new)
probabilities = rf_model.predict_proba(X_new)[:, 1]

# Add to dataframe
new_data['Prediction'] = predictions
new_data['Open_Probability'] = probabilities
new_data['Target_Customer'] = (probabilities >= 0.4).astype(int)  # Adjust threshold

print(new_data[['Gender', 'AGE', 'Open_Probability', 'Target_Customer']].head())
```

---

## 📊 Visualizations Generated

### 1. Decision Tree Analysis
**Files:** `07_decision_tree_feature_importance.png`, `08_decision_tree_structure.png`, `09_decision_tree_confusion_matrix.png`

- Full tree structure visualization (top 3 levels for readability)
- Feature importance bar chart showing most predictive variables
- Confusion matrix with count annotations

### 2. Random Forest Analysis
**Files:** `10_random_forest_feature_importance.png`, `11_random_forest_confusion_matrix.png`

- Feature importance averaged across 100 trees
- Confusion matrix on full dataset

### 3. Neural Network Analysis
**Files:** `13_neural_network_loss_curve.png`, `14_neural_network_confusion_matrix.png`

- Training loss convergence curve
- Final epoch performance visualization

### 4. Model Comparison
**Files:** `12_roc_curve_dt_vs_rf.png`, `15_comprehensive_model_comparison.png`

- Side-by-side ROC curves for all models
- 4-panel comprehensive comparison:
  - ROC curves overlay
  - AUC bar chart
  - Radar chart (all metrics)
  - Grouped bar chart

### 5. Cross-Validation Analysis
**File:** `16_cross_validation_comparison.png`

- Training vs CV AUC comparison
- Overfitting gap visualization
- Shows which model generalizes best

---

## 🔍 Key Findings & Insights

### 1. Feature Importance Rankings

**Decision Tree Top 3:**
1. Bill_Service (most important)
2. Recency_Service
3. Bill_Product

**Random Forest Top 3:**
1. Bill_Service (most important)
2. Recency_Product
3. Bill_Product

**Key Insight:** Billing amounts (especially services) are the strongest predictors of email engagement.

---

### 2. Model Characteristics

| Aspect | Decision Tree | Random Forest | Neural Network |
|--------|---------------|---------------|----------------|
| **Interpretability** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Good | ⭐⭐ Limited |
| **Performance** | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Very Good |
| **Training Speed** | ⭐⭐⭐⭐⭐ Fast | ⭐⭐⭐ Moderate | ⭐⭐ Slow |
| **Overfitting Risk** | ⭐⭐⭐⭐ High | ⭐⭐ Low | ⭐⭐⭐ Moderate |
| **Hyperparameter Tuning** | ⭐⭐⭐ Easy | ⭐⭐⭐ Easy | ⭐ Complex |

---

### 3. Business Recommendations

**Targeting Strategy:**
- Focus campaigns on customers with:
  - ✅ High service billing (>$15)
  - ✅ Recent service purchases (<7 days)
  - ✅ High product billing (>$2)
  - ✅ Recent product purchases (<10 days)

**Probability Threshold Selection:**
- **Conservative (0.6):** 59% precision, 28% recall → Focus on high-confidence customers
- **Balanced (0.4):** ~50% precision, ~45% recall → Broader targeting
- **Aggressive (0.3):** ~40% precision, ~60% recall → Maximize reach

**Expected ROI:**
- For 1,000-email campaign targeting threshold 0.4:
  - Expected opens: ~110-130 (vs 240 with random targeting)
  - Wasted sends: ~370-390
  - Cost efficiency: ~45% better than random

---

## 💡 Technical Insights

### Decision Tree
**Pros:**
- Highly interpretable decision rules
- No feature scaling required
- Handles non-linear relationships
- Fast prediction

**Cons:**
- High variance (small data changes = big tree changes)
- Prone to overfitting
- Biased toward features with many levels

**Best Use Case:** When interpretability is critical and stakeholders need clear decision rules

---

### Random Forest
**Pros:**
- Best predictive performance
- Reduces overfitting through ensemble
- Robust to outliers and noise
- Provides feature importance
- Handles imbalanced data well

**Cons:**
- Less interpretable than single tree
- Slower training and prediction
- Requires more memory

**Best Use Case:** Production deployment where accuracy is priority

---

### Neural Network
**Pros:**
- Can learn complex patterns
- Flexible architecture
- Scales to large datasets
- Good for feature engineering

**Cons:**
- Black-box model (hard to interpret)
- Requires feature scaling
- Hyperparameter sensitive
- Risk of overfitting on small data
- Slower training

**Best Use Case:** Large datasets (>10,000 samples) with complex relationships

---

## 📌 Important Notes

### ⚠️ About the Training Approach

This analysis follows assignment requirements to **train on the entire dataset without train/test split**. This means:

- ✅ **Training metrics** (AUC ~0.96-0.99) are **overly optimistic**
- ✅ **Cross-validation metrics** (AUC ~0.74-0.78) are **realistic estimates**
- ✅ True production performance will be closer to CV results
- ✅ All models show **significant overfitting** (gap ~0.20-0.22)

**For Production:**
- Use proper train/test split or time-based validation
- Collect more data to improve generalization
- Implement monitoring to detect model drift
- Consider ensemble methods or threshold tuning

---

### 🎯 Sensitivity Analysis

**Question 1 Requirement:** Decision Tree sensitivity/recall at 0.50 cutoff

**Result:** Sensitivity = ~0.89 (89%)
- Out of all customers who actually opened emails
- Model correctly identifies 89% at 0.50 probability threshold
- This is on training data (overly optimistic)
- CV estimate: ~0.60-0.65 (more realistic)

**Business Translation:**
- Training result: Can find 9 out of 10 openers
- Realistic result: Can find 6 out of 10 openers
- Use CV result for business planning

---

## 📖 References

- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
- Quinlan, J.R. (1986). *Induction of Decision Trees*. Machine Learning 1, 81-106.
- Rumelhart, D.E., Hinton, G.E., Williams, R.J. (1986). *Learning representations by back-propagating errors*. Nature, 323(6088), 533-536.
- scikit-learn documentation: [Decision Trees](https://scikit-learn.org/stable/modules/tree.html), [Random Forests](https://scikit-learn.org/stable/modules/ensemble.html#forest), [Neural Networks](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830.

---

## 👨‍💼 Author
**William C. Phiri**  
📧 [wphiri@beda.ie]  
🔗 [LinkedIn](https://www.linkedin.com/in/william-phiri-866b8443/)  
🐙 [GitHub: Kochezz](https://github.com/kochezz)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset provided for academic coursework
- Built as part of my Postgraduate Diploma in Data Science Machine Learning module
- Thanks to instructors and peers for feedback
- Inspired by best practices from scikit-learn documentation

---

## 🎓 Key Learning Outcomes

✅ Implemented three distinct ML algorithms from scratch  
✅ Compared tree-based vs neural network approaches  
✅ Performed sensitivity analysis with threshold tuning  
✅ Used cross-validation for realistic performance estimates  
✅ Created production-ready models with persistence  
✅ Generated comprehensive visualizations for stakeholder communication  
✅ Understood overfitting detection and mitigation strategies  
✅ Applied ROC-AUC evaluation for imbalanced classification  

---

**⭐ If you found this project helpful, please consider giving it a star!**

---

## 📝 Appendix: Running Specific Analyses

### A. Feature Importance Only
```python
import joblib
import matplotlib.pyplot as plt

# Load model
rf = joblib.load('models/random_forest_model.pkl')

# Get feature importance
importance = pd.DataFrame({
    'Feature': ['Gender', 'AGE', 'Recency_Service', 'Recency_Product', 
                'Bill_Service', 'Bill_Product'],
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance)
```

### B. Confusion Matrix Analysis
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load model and data
model = joblib.load('models/random_forest_model.pkl')
# ... load X and y ...

# Predict
y_pred = model.predict(X)

# Confusion matrix
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

### C. Custom Threshold Tuning
```python
from sklearn.metrics import precision_recall_curve

# Get probabilities
y_prob = model.predict_proba(X)[:, 1]

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y, y_prob)

# Find optimal threshold (e.g., maximize F1)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"At this threshold: Precision={precision[optimal_idx]:.3f}, Recall={recall[optimal_idx]:.3f}")
```

---

**End of Documentation**

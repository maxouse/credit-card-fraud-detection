# credit-card-fraud-detection
Detecting fraudulent credit card transactions using Machine Learning
# Credit card fraud detection project 

## Dataset

The data used comes from the "Credit Card Fraud Detection" dataset available on Kaggle: [(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)]

**Key characteristics:**
* Contains transactions made by European cardholders in September 2013.
* Features `V1` through `V28` are the result of a **PCA transformation** due to confidentiality.
* Features `Time` (seconds elapsed) and `Amount` (transaction amount) are not PCA-transformed.
* The target variable `Class` is highly imbalanced: fraudulent transactions (`Class=1`) represent only about **0.17%** of the data.

## Methodology 

1.  **Exploratory Data Analysis :**
    * Checked for missing values (none found).
    * Visualized the extreme class imbalance using `countplot`.
    * Confirmed the decorrelation of PCA features (`V1`-`V28`) using a correlation matrix heatmap.

2.  **Preprocessing:**
    * Scaled the `Time` and `Amount` columns using `StandardScaler` to bring them to a similar scale as the PCA features.
    * Split the data into training (25%) and validation (75%) sets using `train_test_split` with a fixed `random_state` for reproducibility. *Note: A 25/75 split surprisingly provided better results than an 80/20 split in cross-validation.*

3.  **Handling Class Imbalance:**
    * Established a model (`LogisticRegression`) on the raw imbalanced data, demonstrating poor recall for the fraud class.
    * Tested **Random UnderSampling**, which improved recall significantly but drastically lowered precision due to information loss.
    * Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic fraud samples and balance the *training data only*.

4.  **Model Training & Evaluation:**
    * Trained `LogisticRegression` on SMOTE data, showing improved recall but still very low precision, indicating the model's linearity was insufficient.
    * Trained tree-based models (`RandomForestClassifier`, `XGBClassifier`, `LGBMClassifier`) on the SMOTE-balanced training data. These models showed a much better balance between precision and recall.
    * Used **5-Fold Cross-Validation** with an imblearn **Pipeline** (SMOTE + LGBMClassifier) on the original (imbalanced) training data to get a robust estimate of the final model's performance and avoid data leakage.

## Key Results 

* The **SMOTE + LGBMClassifier** pipeline achieved the best and most reliable performance during cross-validation (using the 25% training data split):
    * **Average F1-Score (Fraud Class): 0.8049**
    * **Average Recall (Fraud Class): 0.8385** (Catches ~84% of actual frauds)
    * **Average Precision (Fraud Class): 0.7776** (When it predicts fraud, it's correct ~78% of the time)
* **Feature Importance Analysis** (using the final LGBM model) consistently highlighted features like `V14`, `V4`, `V11`, `V5`, and `V12` as the most influential predictors for detecting fraud. 


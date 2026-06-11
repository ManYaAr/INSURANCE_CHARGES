# Insurance Cost Prediction

This notebook explores an insurance dataset to predict individual medical costs. It performs exploratory data analysis (EDA), preprocesses the data, trains multiple regression models, and evaluates their performance.

## Dataset

The dataset contains information about individuals including:
- **age**: age of primary beneficiary
- **sex**: insurance contractor gender, female or male
- **bmi**: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m^2) using the ratio of weight to square of height
- **children**: Number of children covered by health insurance / Number of dependents
- **smoker**: Smoker or not
- **region**: the beneficiary's residential area in the northeast, southeast, southwest, northwest
- **charges**: Individual medical costs billed by health insurance

## Exploratory Data Analysis (EDA)

The notebook includes comprehensive visualizations to understand the relationships between different features and the target variable ('charges'):

- Scatter plot of charges by age and sex
- Linear regression plot of charges by age, smoker status, and sex
- Boxplot of charges by region and sex
- Boxplot of charges by number of children and smoker status
- Countplot of sex by smoker status
- Histogram of age distribution with KDE

### Key Insights from EDA
- Age and smoker status have strong positive correlations with charges (0.534 and 0.663 respectively)
- Number of children also shows a positive correlation with charges (0.133)
- Sex and region have minimal correlation with charges

## Data Preprocessing

### Feature Engineering
- **smoker_bmi**: Interaction feature (smoker indicator × BMI)
- **smoker_age**: Interaction feature (smoker indicator × age)

### Encoding Strategy
- **Numerical features** (age, bmi, children, smoker_bmi, smoker_age): StandardScaler normalization
- **region**: TargetEncoder (categorical to numerical based on target mean)
- **smoker, sex**: OneHotEncoder (categorical features)

### Train-Test Split
- Training set: 80%
- Testing set: 20%
- Random state: 42

## Model Training

Four regression models are trained and evaluated:
1. **Random Forest Regressor** - Ensemble method with decision trees
2. **Decision Tree Regressor** - Single tree-based model
3. **Gradient Boosting Regressor** - Sequential ensemble method
4. **XGBoost Regressor** - Extreme Gradient Boosting

All models are trained within a scikit-learn Pipeline for reproducibility and proper preprocessing.

## Model Evaluation

Models are evaluated using cross-validation (5-fold KFold) with the following metrics:
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual values
- **Mean Squared Error (MSE)**: Average squared differences
- **Root Mean Squared Error (RMSE)**: Square root of MSE
- **Mean Absolute Percentage Error (MAPE)**: Average percentage error
- **R-squared**: Coefficient of determination

## Results

Cross-validation results show:
- **Random Forest Regressor**: MAE ≈ $2,702.42
- **Gradient Boosting Regressor**: MAE ≈ $2,683.45 (Best performer)
- **Decision Tree Regressor**: MAE ≈ $3,162.45
- **XGBoost Regressor**: MAE ≈ $3,257.37

The Gradient Boosting Regressor provides the most accurate predictions.

## Visualization of Results

- Residual plots to assess model assumptions
- Histograms of residuals to check for normality
- Scatter plots of actual vs. predicted charges to visualize predictive accuracy
- Model comparison using cross-validation metrics

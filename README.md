# Insurance Cost Prediction

This notebook explores an insurance dataset to predict individual medical costs. It performs exploratory data analysis (EDA), preprocesses the data, trains two regression models (Gradient Boosting Regressor and XGBoost Regressor), and evaluates their performance.

## Dataset

The dataset contains information about individuals including:
- age: age of primary beneficiary
- sex: insurance contractor gender, female or male
- bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m^2) using the ratio of weight to square of height, ideally 18.5 to 24.9
- children: Number of children covered by health insurance / Number of dependents
- smoker: Smoker or not
- region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
- charges: Individual medical costs billed by health insurance

## Exploratory Data Analysis (EDA)

The notebook includes visualizations to understand the relationships between different features and the target variable ('charges').

- Scatter plot of charges by age and sex.
- LM plot of charges by age, smoker status, and sex.
- Boxplot of charges by region and sex.
- Countplot of sex by smoker status.
- Histogram of age distribution.

## Data Preprocessing

- Categorical features are converted into numerical format using one-hot encoding.
- The data is split into training and testing sets (80% training, 20% testing).

## Model Training

Two regression models are trained:
- Gradient Boosting Regressor
- XGBoost Regressor

## Model Evaluation

The models are evaluated using the following metrics on both the training and testing sets:
- R-squared Error
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

## Results

The evaluation metrics for both models on the training and test sets are presented, allowing for comparison of their performance.

## Visualization of Results

- Residual plots for the Gradient Boosting Regressor to assess model assumptions.
- Histograms of residuals to check for normality.
- Scatter plots of actual vs. predicted charges for both models to visualize their predictive accuracy.

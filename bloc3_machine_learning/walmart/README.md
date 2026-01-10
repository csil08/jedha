# Walmart weekly sales prediction

This project aims to **identify the influential drivers of weekly sales of Walmart** and make **sales predictions**, using a **machine learning model**.

## Dataset

The dataset used for this project is an *unbalanced panel** dataset of 150 observations about weekly sales achieved by 20 different Walmart stores over a period between 05/02/2010 to 19/10/2012. The dataset also contains data about 
exogenous variables (such as unemployment rate, fuel price, temperature, Consumer Price Index) which might be useful for predicting the amount of sales.

## Project workflow

The analysis follows these steps:

- **Initial exploration and overview of the dataset**
    - missing values, duplicates

- **Preprocessings** to prepare data for modelling

- **EDA and prior selection of the variables**

- **Model training**:
    - Baseline model: **linear regression model**
    - Training **regularized regression models (Ridge, Lasso)** to avoid overfitting, with hyperparameter tuning via GridSearchCV


## Deliverables

Jupyter notebook (available in the repository): `walmart.ipynb`


## Tech stack
| **Category**       | **Technology / Library**                  |
|---------------------|------------------------------------------|
| Programming language           | Python                       |
| Data processing                 | Pandas, NumPy               |
| Data visualisation    | Plotly                                |
| Machine learning      | Scikit-learn                          |


## Key insights

### Models' performance
| Model               | R² (Train) | R² (Test) | RMSE (Train) | RMSE (Test) | MAE (Train) | MAE (Test) |
|---------------------|------------|-----------|--------------|-------------|-------------|------------|
| Linear regression          | 0.986      | 0.972      | 0.08         | 0.115        | 0.061        | 0.101       |
| Ridge regression ($\alpha$ = 0.05)          | 0.985       | 0.976      | 0.082         | 0.107        | 0.061        | 0.09       |
| Lasso regression ($\alpha$ = 0.001)        | 0.983       | 0.98      | 0.088         | 0.097        | 0.065        | 0.079       |

*NB: RMSE and MAE are expressed in millions USD.*

### Feature importance
The store-specific effects have a higher impact on sales, in contrast to exogeneous variables such as CPI, unemployment rate, fuel price and temperature.  
Seasonality also plays a role, in particular December shows a strong positive influence on sales, due to Christmas and end-of-year holidays and celebrations.

Example for ridge regression (hyperparameter tuned using GridSearch CV):
![Ridge regression - feature importance](img/ridge_gscv_feature_importance.png)

### Methodological limits
The modelling does not properly takes into account the temporal structure of the dataset. It suffers from **data leakage** because of improper data splitting (where future data is included in the training set), which may lead to overestimate the models' performance.

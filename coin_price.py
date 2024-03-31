# Import necessary libraries
import numpy as np
import pandas as pd
# from sklearn.base import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the dataset
data = pd.read_csv('heated1.csv')  # Replace 'your_dataset.csv' with the actual filename

# Split the data into independent variables (X) and the target variable (y)
X = data.drop('current_price_usd', axis=1)  # Assuming 'coin_price' is the target variable
y = data['current_price_usd']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict coin prices on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

# Optionally, you can also print the coefficients and intercept of the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)


from sklearn.ensemble import RandomForestRegressor

# Create and fit the Random Forest regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust the number of trees (n_estimators)
rf_model.fit(X_train, y_train)

# Predict coin prices on the test set
rf_y_pred = rf_model.predict(X_test)

# Evaluate the model
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_mae = mean_absolute_error(y_test, rf_y_pred)
# rf_r2 = r2_score(y_test, rf_y_pred)

print("Random Forest Mean Squared Error:", rf_mse)
print("Random Forest Mean Absolute Error:", rf_mae)
# print("Random Forest R-squared:", rf_r2)


from joblib import dump
# Save the Random Forest model as a .pkl file
dump(rf_model, 'model.pkl')






# import pandas as pd
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import RegexpTokenizer
# import re
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LinearRegression, Lasso
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestRegressor
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle

# # %matplotlib inline
# # plt.style.use('fivethirtyeight')

# # Save the Random Forest model as a .pkl file
# # dump(rf_model, 'random_forest_model3.pkl')
# def basic_linear_model(X, y, model, **kwargs):
#     """basic linear model that takes in features and response variable,
#     prints r-squared, graphs predicted vs. y-test, and returns predicted and actual
#     test values"""
#     lr = model(**kwargs)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#     lr.fit(X_train, y_train)
#     print('The R-squared score: ', lr.score(X_test, y_test))
#     preds = lr.predict(X_test)
#     fig, ax = plt.subplots(figsize=(15,7))
#     plt.scatter(np.arange(len(preds)), np.sort(preds), alpha=0.5, c='r', label='predictions')
#     plt.scatter(np.arange(len(preds)), np.sort(y_test), alpha=0.5, c='g', label='true values')
#     plt.legend().set_alpha(1)
#     plt.show()

#     return y_test, preds

# romanh=pd.read_csv('heated1.csv')

# y = np.log(romanh['current_price_usd'].values)
# X = romanh.drop([ 'current_price_usd'], axis=1)

# y_test, preds = basic_linear_model(X, y, RandomForestRegressor)


# from sklearn.ensemble import RandomForestRegressor
# import pickle

# # Assuming X_train and y_train are your training data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# # Train the RandomForestRegressor model
# random_forest = RandomForestRegressor()
# random_forest.fit(X_train, y_train)

# # Save the trained model to a file named 'random_forest_model.pkl'
# with open('random_forest_model.pkl', 'wb') as f:
#     pickle.dump(random_forest, f)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# param_grid = {"n_estimators": [200, 500],
#     "max_depth": [3, None],
#     "max_features": [1, 3, 5, 10],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 3, 10],
#     "bootstrap": [True, False]}

# model = RandomForestRegressor(random_state=0)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
# grid.fit(X_train, y_train)

# print(grid.best_score_)
# print(grid.best_params_)


# from catboost import CatBoostRegressor
# y_test, preds = basic_linear_model(X, y, CatBoostRegressor) 

# from lightgbm import LGBMRegressor
# y_test, preds = basic_linear_model(X, y, LGBMRegressor)

# from joblib import dump

# # Save the Random Forest model as a .pkl file
# dump(basic_linear_model, 'model.pkl')
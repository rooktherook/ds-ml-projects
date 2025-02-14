from azureml.core import Run
import pandas as pd
import numpy as np
import joblib
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

run = Run.get_context()

data = pd.read_csv('salaries_processed.csv')


Y = data['salary_in_usd']
X = data.drop('salary_in_usd', axis=1)

# Using a 80-20 train-test split for the training and testing set
# Test size 0.1 and 0.3 were tested to be less optimal
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


# Train model
model = LinearRegression()
model.fit(X_train, Y_train)


# Predict the test set
Y_pred = model.predict(X_test)

# Log the root mean squared error.
mse = mean_squared_error(Y_test, Y_pred, squared=False)
run.log('Root Mean Squared Error', np.float(mse))

# Save the trained model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')


run.complete()
# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2. Data preparation
3. Hypothesis Definition
4. Cost Function
5. Parameter Update Rule
6. Iterative Training
7. Model evaluation
8. End

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Imesh Karthick P R
RegisterNumber: 212225060092 
*/
```

import numpy as np

from sklearn.datasets import fetch_california_housing

from sklearn.linear_model import SGDRegressor

from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()

X = data.data[:, :3]

Y = np.column_stack((data.target, data.data[:, 6]))

print("X shape:", X.shape)

print("Y shape:", Y.shape)

print("Example X (first row):", X[0])

print("Example Y (first row):", Y[0])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



print("Train shapes:", X_train.shape, Y_train.shape)

print("Test shapes: ", X_test.shape, Y_test.shape)



scaler_X = StandardScaler()

scaler_Y = StandardScaler()



X_train_scaled = scaler_X.fit_transform(X_train)

X_test_scaled = scaler_X.transform(X_test)



Y_train_scaled = scaler_Y.fit_transform(Y_train)

Y_test_scaled = scaler_Y.transform(Y_test)



print("Scaled X_train mean (approx):", X_train_scaled.mean(axis=0))

print("Scaled Y_train mean (approx):", Y_train_scaled.mean(axis=0))

sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)  # you can also set alpha, eta0, penalty etc.

multi_output_sgd = MultiOutputRegressor(sgd)

multi_output_sgd.fit(X_train_scaled, Y_train_scaled)

Y_pred_scaled = multi_output_sgd.predict(X_test_scaled)   # predicted in scaled space

Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)         # back to original units

Y_test_orig = scaler_Y.inverse_transform(Y_test_scaled)    # ground-truth back to original

print("First 5 predictions (original units):")

print(Y_pred[:5])

mse = mean_squared_error(Y_test_orig, Y_pred)

print("Mean Squared Error (multi-output):", mse)

mse_per_output = np.mean((Y_test_orig - Y_pred) ** 2, axis=0)

print("MSE per output:", mse_per_output)

for i in range(5):

    print(f"Example {i+1}")
    
    print("Inputs (raw):", X_test[i])
    
    print("True outputs:", Y_test_orig[i])
    
    print("Predicted   :", Y_pred[i])
    
    print("-" * 40)


from sklearn.linear_model import LinearRegression, SGDRegressor

from sklearn.metrics import mean_squared_error

import numpy as np

from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()

X, y = data.data[:, :3], data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


lr = LinearRegression()

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01, learning_rate='constant', random_state=42)

sgd.fit(X_train, y_train)

sgd_pred = sgd.predict(X_test)

print("LinearRegression MSE:", mean_squared_error(y_test, lr_pred))


print("SGDRegressor MSE:", mean_squared_error(y_test, sgd_pred))

## Output:
<img width="697" height="647" alt="image" src="https://github.com/user-attachments/assets/3f890a6a-9d66-4bc1-a096-0a7a6196a8c5" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.

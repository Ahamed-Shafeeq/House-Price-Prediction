import pandas as pd
import numpy as np  
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Read the House data using pandas.
data = pd.read_csv("DataSet/House_Data.csv")


data.drop("Id", axis=1, inplace=True) 
'''Id is not required for house prediction because it is just a serial number.
    Model can confuse by it. So droppping it can increase accuracy.
'''

#x = input. Input should be [Area, Bedrooms, Bathrooms...etc(excluding Price)].
x = data.drop("Price", axis=1) 

#y = output. Output should only be a Price Because We are going predict the House price based on input.
y = data["Price"] 


#Splitting the data into two groups Numerical and Categorical 
numerical_features = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"]
categorical_features = ["Location", "Condition", "Garage"]

#Scale the numerical data and Encode the categorical data.
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")

#Preprocessing the data.
preprocessor_data = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

model = Pipeline(
    steps=[("preprocessor", preprocessor_data),
    ("regressor", LinearRegression())]
)

#Split the data into two. One for model training and one model testing.
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#Training the model.
model.fit(X_train, Y_train)#Models learns from input and predict the output.

#Prediction of Output.
Y_prediction = model.predict(X_test)

#Finding the mean absolute error(MSE) and mean squared error(MSE).It will predict the average difference between real and predicted price.
#MAE Formula->Average of (actualprice - predictedprice).
#MSE Formula->Average of (actual_price − predicted_price)².
mae = mean_absolute_error(Y_test, Y_prediction)
mse = mean_squared_error(Y_test, Y_prediction)#MSE punishes the large errors more heavily.

#Finding the root mean squared error(RMSE).
rmse = np.sqrt(mse)

#Finding the r2_Score.It will tell that how much better is my model than guessing the average house price.
r2_Score = r2_score(Y_test, Y_prediction)

print("\nModel Performance:")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2_Score:.2f}")


with open("Model/House_Price_Model.pkl", "wb") as f:
    pickle.dump(model, f)

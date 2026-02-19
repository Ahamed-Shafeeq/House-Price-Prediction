# 🏠 House Price Prediction Web App

A machine learning web application that predicts house prices based on user input, built using Python and Streamlit.

## 📌 Project Overview

This project is an end-to-end machine learning application that predicts house prices using a Linear Regression model. 
Users can enter house-related features through a Streamlit web interface and instantly receive a predicted price.

## 🎯 Motivation & Learning

I may not have trained this model perfectly, but I did my best using my current understanding of Linear Regression. 
This project was built by implementing concepts learned from the Coursera course 
**"IBM Machine Learning with Python"**.

In the future, I plan to improve this project by experimenting with other machine learning algorithms 
to achieve better prediction performance.

## 🛠 Tech Stack

numpy
pandas
matplotlib
seaborn
scikit-learn
streamlit
joblib


## 📂 Project Structure

HousePricePrediction/
├── DataSet/
│   └── House_Data.csv          # Dataset used for training the model
│
├── Model/
│   └── House_Price_Model.pkl   # Trained machine learning model
│
├── src/
│   └── Model_Training.py       # Model training & preprocessing logic
│
├── env/                        # Virtual environment (ignored in GitHub)
│
├── App.py                      # Streamlit web application
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Files & folders ignored by Git

This structure keeps the dataset, model, training logic, and application code organized and easy to maintain.


## ▶️ How to Run

### 1. Clone the repository

```bash

git clone https://github.com/Ahamed-Shafeeq/House-Price-Prediction.git
cd HousePricePrediction
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
streamlit run app.py



## 📊 Model Performance

Model Performance:
MAE  : 243241.98
MSE  : 78321466146.03
RMSE : 279859.73
R²   : -0.01

These results indicate that the model has room for improvement, which will be addressed in future versions.

## 🚀 Future Improvements

- Try advanced models (Random Forest, XGBoost)
- Feature engineering
- Hyperparameter tuning
- Model deployment improvements

## 👤 Author

**Ahamed Shafeeq**  
Aspiring AI/Machine Learning Engineer.

🔗 GitHub: https://github.com/Ahamed-Shafeeq  
🔗 LinkedIn: https://www.linkedin.com/in/ahamed-shafeeq-822419380





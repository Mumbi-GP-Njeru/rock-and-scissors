# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load the data
url = 'https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv'
data = pd.read_csv(url)

# Split the data into training and test datasets
train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=42)

# Extract labels
train_labels = train_dataset.pop('median_house_value')
test_labels = test_dataset.pop('median_house_value')

# Create the model and train it
model = LinearRegression()
model.fit(train_dataset, train_labels)

# Check the model's performance on the test dataset
predicted_expenses = model.predict(test_dataset)
mae = mean_absolute_error(test_labels, predicted_expenses)
print('Mean Absolute Error:', mae)

# Check the model's performance on the train dataset
predicted_expenses_train = model.predict(train_dataset)
mae_train = mean_absolute_error(train_labels, predicted_expenses_train)
print('Mean Absolute Error on train data:', mae_train)

# Visualize the results
plt.scatter(test_labels, predicted_expenses)
plt.xlabel('True Expenses')
plt.ylabel('Predicted Expenses')
plt.title('Linear Regression - Expenses vs Predicted Expenses')
plt.show()
# ML Test on Prices of Laptops using scikit-learn

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

print("Initiliazing path to read data...")
laptop_data_path = "laptops.csv"
print("Reading csv files using pandas. Encoding used: \"ISO-8859-1\"")
print("Encoding is used so that we are able to read the CSV file, due to how the read_csv handles the input file.")
# For more info, please refer to the link below:
# https://stackoverflow.com/questions/18171739/unicodedecodeerror-when-reading-csv-file-in-pandas-with-python
laptop_data = pd.read_csv(laptop_data_path, encoding = "ISO-8859-1")
print("Finished reading csv file...\n")

print("Printing the first 5 data for viewing...")
data_head = laptop_data.head()
print(data_head)
print("\n")

print("Printing the available columns in the data...")
cols = laptop_data.columns
print(cols)

print("Let's drop the missing values from our data.")
laptop_data = laptop_data.dropna(axis=0)

print("Then, we will pick a Prediction Target. In this case, we will pick the price.")
y = laptop_data.Price_euros

print("After that, we will pick the Features that we want that will help to weigh in, in the selection process.")
# You need to be careful on what features you are expecting. Normally, they can only extract numbers. I am sure there is a way to if we can numerize the brand to make it more ML-friendly data.
laptop_features = ['Inches', 'Ram', 'Weight']
X = laptop_data[laptop_features]

print("Let's see what the Feature is all about by using the describe function.")
X.describe()

print("And let's see what's in there for a little bit")
print(X.head())

print ("Now, we create our model for the Machine Learning")
laptop_ml = DecisionTreeRegressor(random_state=1)

print("Next, we fit the model using X and y")
laptop_ml.fit(X,y)

print("Making predictions for the following 5 laptops:")
print(X.head())
print("The predictions are")
print(laptop_ml.predict(X.head()))

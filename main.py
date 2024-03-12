# We import the libraries we need.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Read the dataset.
data = pd.read_csv("Ice Cream Sales - temperatures.csv")

# DATA ANALYSIS

# Display the first few rows of the data.
print("First few rows of the data: ", data.head())

# Learning Columns.
print("Learning Columns: ", data.columns)

# Learning Datatypes of the Columns.
print("Learning Datatypes of the Columns:\n", data.dtypes)

# Learning shape of the dataset.
print("Learning shape of the dataset: ", data.shape)

# Check for missing values.
print("Check for missing values:\n", data.isnull().sum())

# Summary statistics.
print("Summary Statistisc:\n", data.describe())

# DATA VISUALIZATION

sns.scatterplot(x="Ice Cream Profits", y="Temperature", data=data, color="lightblue")
plt.title("Ice Cream Profits - Temperature")
plt.xlabel("Ice Cream Profits")
plt.ylabel("Temperature")
plt.show()

# LINEAR REGRESSION

X = data[["Temperature"]]
y = data[["Ice Cream Profits"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# VISUALIZATION OF THE RESULT

plt.scatter(X_train, y_train, color="pink")
X_train_pred = model.predict((X_train))
plt.scatter(X_train, X_train_pred, color="lightblue")
plt.title('Ice Cream Profits - Temperature')
plt.xlabel('Ice Cream Profits')
plt.ylabel('Temperature')
plt.show()

# SUCCESS RATE OF THE MODEL

print("R2 Score: ", r2_score(y_test, y_pred)*100)
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))

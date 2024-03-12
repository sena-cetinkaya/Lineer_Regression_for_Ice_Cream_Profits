# Lineer Regression for Ice Cream Profits
This repo contains an example of a Linear Regression model created using "Temperature" and "Ice Cream Profits" values. The dataset used in this repo was taken from Kaggle. [Link to the dataset.](https://www.kaggle.com/datasets/raphaelmanayon/temperature-and-ice-cream-sales)

To review my work on this dataset on Kaggle; [https://www.kaggle.com/code/senacetinkaya/linear-regression-for-ice-cream-profits](https://www.kaggle.com/code/senacetinkaya/linear-regression-for-ice-cream-profits)

------------------------------
## LINEAR REGRESSION
Linear regression analysis is used to predict the value of a variable based on the value of another variable. The variable you want to predict is called the dependent variable. The variable you are using to predict the other variable's value is called the independent variable. This form of analysis estimates the coefficients of the linear equation, involving one or more independent variables that best predict the value of the dependent variable.

This repo contains a Linear Regression model that predicts "Ice Cream Profits" values ​​from "Temperature" values. R-square, mean absolute error and mean squared error metrics were used for the success rate of the model.

------------------------------
### Libraries Used in the Project
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
```

------------------------------
### Data Visualization
Let's observe the relationship between Ice Cream Profits data and Temperature.

![](https://github.com/sena-cetinkaya/Lineer_Regression_for_Ice_Cream_Profits/blob/main/Figure_1.png)

### Visualization Of the Result
Pink colored values are the values of real data. The ones in blue are the values predicted by the model.

![](https://github.com/sena-cetinkaya/Lineer_Regression_for_Ice_Cream_Profits/blob/main/Figure_2.png)

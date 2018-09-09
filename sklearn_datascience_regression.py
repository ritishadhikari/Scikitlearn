import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error #for finding out the least squares method


df1 = pd.read_csv(r"C:\Users\ritis\PycharmProjects\CSV Files\Advertising_scikit.csv", index_col='Unnamed: 0')
print("The Original DataFrame Looks Like \n", df1, "\n")
print("The Shape of the Data Frame is \n", df1.shape, "\n")

sns.pairplot(data=df1,x_vars=['TV','Radio','Newspaper'],y_vars=['Sales'],height=3,kind='reg')

X = df1[['TV','Radio','Newspaper']]
y = df1['Sales']
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.25,random_state=1)

lng = LinearRegression()
print("The Input Values of Linear Regression are :\n", lng, "\n")
print("The Output Parameters of the Linear Regression are :\n", dir(lng), "\n")
lng.fit(Xtrain,ytrain)
print("Coeficients are :\n ",lng.coef_, "\n")
print("Intercepts are :\n", lng.intercept_, "\n")

dict ={}
p =0
for i in X:
    dict[i] = lng.coef_[p]
    p+=1
print("The Coeficant Values for the Attributes are :\n", dict, "\n")

ypredict = lng.predict(Xtest)
print("Prediction :\n", ypredict, "\n")

MSE = mean_squared_error(ytest,ypredict)
print("The Mean Squared Error is : \n", MSE, "\n")
print("The Root Mean Squared Error is : \n", np.sqrt(MSE), "\n")

m=[[160,80,40]]
value = lng.predict(m)
print("Value :\n",value, "\n")

print("Now finding the Errors without the NewsPaper attribute as it's coefficient value is very less : \n")

X = df1[['TV','Radio']]
y = df1['Sales']
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.25,random_state=1)

lng = LinearRegression()
print("The Input Values of Linear Regression are :\n", lng, "\n")
print("The Output Parameters of the Linear Regression are :\n", dir(lng), "\n")
lng.fit(Xtrain,ytrain)
print("Coeficients are :\n ",lng.coef_, "\n")
print("Intercepts are :\n", lng.intercept_, "\n")

dict ={}
p =0
for i in X:
    dict[i] = lng.coef_[p]
    p+=1
print("The Coeficant Values for the Attributes are :\n", dict, "\n")

ypredict = lng.predict(Xtest)
print("Prediction :\n", ypredict, "\n")

MSE = mean_squared_error(ytest,ypredict)
print("The Mean Squared Error is : \n", MSE, "\n")
print("The Root Mean Squared Error is : \n", np.sqrt(MSE), "\n") #Lowest RMSE is the best



plt.show()



from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target
knn = KNeighborsClassifier(n_neighbors=5)
score = []
for i in range (1,11):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.1,random_state=i)
    knn.fit(Xtrain,ytrain)
    ypred = knn.predict(Xtest)
    accuracy = accuracy_score(ytest,ypred)
    score.append(accuracy)
print("The Average Accuracy for Logistic Regression through Train Test Split varied across Five Random State is :\n",np.mean(score),"\n")

score = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
print("Average Score from the Cross Val Score from knn=5 is :\n",score.mean(), "\n")

average_score = []
average_score_dict ={}
for i in range (1,31):
    knn = KNeighborsClassifier(n_neighbors=i)
    score =cross_val_score(estimator=knn,X=X,y=y,scoring='accuracy',cv=10)
    average_score.append(score.mean())
    average_score_dict[i]=score.mean()
print("The Values for Average Score for Kneighbours from 1 to 20 are :\n", average_score,"\n")
print("The Values for Average Score Dict for Kneighbours from 1 to 20 are :\n", average_score_dict,"\n")
plt.plot(average_score)
#Higher Values of K makes the simplest model

lgg = LogisticRegression()
score = cross_val_score(estimator=lgg,X=X,y=y,scoring='accuracy',cv=10)
print("Average Score for Cross Validation through Logistic Regression is :\n", score.mean(), "\n")

#Working on Cross Val Score for the original DataSets :
df1 = pd.read_csv(r"C:\Users\ritis\PycharmProjects\CSV Files\Advertising_scikit.csv", index_col='Unnamed: 0')
X = df1[['TV','Radio','Newspaper']]
y = df1['Sales']

lng =LinearRegression()
score = cross_val_score(estimator=lng,X=X,y=y,scoring='neg_mean_squared_error',cv=10)
print("Score for Linear Regression for the Advertising Data :\n", score,"\n")
root_mean_square_score = np.mean(np.sqrt(-score))
print("The root mean square score is :\n", root_mean_square_score)



#features excluding newspaper

X = df1[['TV','Radio']]
y = df1['Sales']

lng =LinearRegression()
score = cross_val_score(estimator=lng,X=X,y=y,scoring='neg_mean_squared_error',cv=10)
print("Score for Linear Regression for the Advertising Data :\n", score,"\n")
root_mean_square_score = np.mean(np.sqrt(-score))
print("The root mean square score without Newspaper is :\n", root_mean_square_score)

plt.show()








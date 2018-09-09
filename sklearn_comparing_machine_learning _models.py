import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score #Importing the Accuracy Classifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#Train and Test on the entire Dataset

iris =load_iris()
X= iris.data
y= iris.target

logreg = LogisticRegression()
logreg.fit(X,y)
y_pred = logreg.predict(X)
print("Predicting the Data which have already been Trained through Logistic Regression \n",y_pred,"\n")
print ("Accuracy Score for Logistic Regression for the data already trained on \n",accuracy_score(y,y_pred),"\n")

knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X,y)
y_pred = knn1.predict(X)
print("Predicting the Data which have already been Trained through Knn Classifier with n=1 \n",y_pred,"\n")
print ("Accuracy Score for Knn Classifier with n=1 for the data already trained on \n",accuracy_score(y,y_pred),"\n")

knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X,y)
y_pred = knn5.predict(X)
print("Predicting the Data which have already been Trained through Knn Classifier with n=5 \n",y_pred,"\n")
print ("Accuracy Score for Knn Classifier with n=5 for the data already trained on \n",accuracy_score(y,y_pred),"\n")

#Train Test and Split
X_Train, X_Test, y_Train, y_Test = train_test_split(X,y, train_size=0.60, random_state=4)

logreg.fit(X_Train,y_Train)
y_pred = logreg.predict(X_Test)
print("Predicting the Data which have undergone Train Test and Split through Logistic Regression \n",y_pred, "\n")
print("Accuracy Score for Logistic Regression for Train Test and Split \n", accuracy_score(y_Test,y_pred),"\n")


knn1.fit(X_Train,y_Train)
y_pred = knn1.predict(X_Test)
print("Predicting the Data which have undergone Train Test and Split through knn = 1 \n",y_pred, "\n")
print("Accuracy Score for knn = 1 for Train Test and Split \n", accuracy_score(y_Test,y_pred),"\n")


knn5.fit(X_Train,y_Train)
y_pred = knn5.predict(X_Test)
print("Predicting the Data which have undergone Train Test and Split through knn = 5 \n",y_pred, "\n")
print("Accuracy Score for knn = 5 for Train Test and Split \n", accuracy_score(y_Test,y_pred),"\n")

k = range(1,26)
scores = []
score_dictionary = {}
for i in k:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_Train,y_Train)
    y_pred = knn.predict(X_Test)
    accuracy = accuracy_score(y_Test,y_pred)
    scores.append(accuracy)
    score_dictionary[i]=accuracy

print("Scores :\n ",scores, "\n")
print("Scores Dictionary :\n ",score_dictionary, "\n")

plt.plot(k,scores)
plt.xlabel("Value of K")
plt.ylabel("Value of Accuracy for Normal State = 4")

#Taking knn no 11 from the graph
knn11 = KNeighborsClassifier(n_neighbors=11)
knn11.fit(X_Train,y_Train)
print("Taking Knn Value of 11 from the Graph Generated and predicting an unknown Model :\n",knn11.predict([[3,5,4,2]]),"\n")

plt.show()










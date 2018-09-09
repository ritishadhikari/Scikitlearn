from sklearn.datasets import load_iris #Importing the Dataset
from sklearn.neighbors import KNeighborsClassifier  #Importing the Classifier
from sklearn.linear_model import LogisticRegression

iris = load_iris() #Instantiating the Dataset
X = iris.data
y= iris.target

knn= KNeighborsClassifier(n_neighbors=1) #Instantiating the Classifier by tuning the model with appropriate Parameters
print("Assigned Values of Knn Classifiers are :\n", knn, "\n")
print("Directories of Knn Classifiers are :\n", dir(knn), "\n")

knn.fit(X,y) #Fitting the model with the Data

Prediction1 = knn.predict([[3,5,4,2]]) #Predicting with an unknown Data
print("The Prediction for the new dataset is :\n", Prediction1)

Prediction2 = knn.predict([[3,5,4,2],[5,4,3,2]]) #Predicting with an unknown Data
print("The Prediction for the new datasets are :\n", Prediction2)

#Predicting with another method
lgg =LogisticRegression()
lgg.fit(X,y)

Prediction3 = lgg.predict([[3,5,4,2]]) #Predicting with an unknown Data
print("The Prediction from Logistic Regression for the new dataset is :\n", Prediction3)

Prediction4 = lgg.predict([[3,5,4,2],[5,4,3,2]]) #Predicting with an unknown Data
print("The Prediction from Logistic Regression for the new datasets are :\n", Prediction4)

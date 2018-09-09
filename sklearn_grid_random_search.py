import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.model_selection import RandomizedSearchCV

iris = load_iris()
X = iris.data
y = iris.target
k = np.arange(start=1,step=1,stop=31)
paramgrid = dict(n_neighbors=k)
print("The Range of Values for which Grid Search CV will run is :\n", paramgrid, "\n")

knn = KNeighborsClassifier()

grid = GridSearchCV(estimator=knn,param_grid=paramgrid,scoring='accuracy',cv=10)
print("Grid Contains :\n", grid,"\n")
grid.fit(X=X,y=y)
print("Grid Directory Contains :\n",dir(grid),"\n")

print("Grid Scores for the Knn Algorith is :\n", grid.grid_scores_,"\n")
#print("Grid Scores for the Knn Algorith is :\n",grid.cv_results_,"\n")

#Checking for the first Grid Score Value :
print("The Grid Score contains the following :\n", dir(grid.grid_scores_[0]),"\n")
print("The First Grid Score is :\n", grid.grid_scores_[0],"\n")
print("The First Grid Parameter is :\n", grid.grid_scores_[0].parameters,"\n")
print("The Cross Validation Scores for the First Grid is \n",grid.grid_scores_[0].cv_validation_scores,"\n")
print("The Mean Validation Score for the First Grid is :\n ",grid.grid_scores_[0].mean_validation_score,"\n")

grid_mean_scores =[]
for i in grid.grid_scores_:
    grid_mean_scores.append(i.mean_validation_score)
print("The Grid Mean Scores are :\n",grid_mean_scores)

grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
print("Another way to print the Grid Mean Scores :\n",grid_mean_scores, "\n")

plt.plot(k,grid_mean_scores)
plt.xlabel("Value of K for Grid Search")
plt.ylabel(("Value of the accuracy"))

#Checking for the Best Grid Value
print("The Best Grid Score is :\n", grid.best_score_,"\n")
print("The Best Grid Params is :\n", grid.best_params_,"\n")
print("The Best Grid Estimator is :\n", grid.best_estimator_,"\n")

print("Checking for Multiple Parameters\n")
k = np.arange(start=1,step=1,stop=31)
weights = ['uniform','distance']
paramgrid = dict(n_neighbors=k, weights=weights)
print("The Range of Values for which Grid Search CV will run is :\n", paramgrid, "\n")

grid = GridSearchCV(estimator=knn,param_grid=paramgrid,scoring='accuracy',cv=10)
print("Grid Contains :\n", grid,"\n")
grid.fit(X=X,y=y)
print("Grid Directory Contains :\n",dir(grid),"\n")

print("Grid Scores for the Knn Algorith is :\n", grid.grid_scores_,"\n")
#print("Grid Scores for the Knn Algorith is :\n",grid.cv_results_,"\n")

#Checking for the Best Grid Value
print("The Best Grid Score is :\n", grid.best_score_,"\n")
print("The Best Grid Params is :\n", grid.best_params_,"\n")
print("The Best Grid Estimator is :\n", grid.best_estimator_,"\n")

print("Predicting from the best Grid Response \n", grid.predict([[3,5,4,2]]),"\n")

print("Doing Random Search CV \n")

random =RandomizedSearchCV(estimator=knn,param_distributions=paramgrid,n_iter=10,scoring='accuracy',cv=10,random_state=4)
random.fit(X=X,y=y)
print("Random Directory Contains :\n",dir(random),"\n")

#Checking for the best value:
print("The Best Random Score is :\n", random.best_score_,"\n")
print("The Best Random Params is :\n", random.best_params_,"\n")
print("The Best Random Estimator is :\n", random.best_estimator_,"\n")

print("Predicting from the best Random Response \n", random.predict([[3,5,4,2]]),"\n")

best_score =[]
for i in range(100):
    random =RandomizedSearchCV(estimator=knn,param_distributions=paramgrid,n_iter=10,cv=10,scoring='accuracy')
    random.fit(X=X,y=y)
    best_score.append(random.best_score_)
print("The Best Scores from Random Search out of 100 observations are :\n",best_score, "\n")


#plt.plot(best_score)
#plt.xlabel("Opportunities ")
#plt.ylabel("Accuracy Score")


plt.show()


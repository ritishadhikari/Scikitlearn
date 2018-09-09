from sklearn.datasets import load_iris

iris = load_iris()
print("Type of Iris is :\n", type(iris), "\n")
print("Directories in Iris are:\n", dir(iris), "\n")
print("Data of the Iris Dataset is :\n", iris.data, "\n")
#Rows are also known as Observation, Sample, Instance or Record
#Columns are also known as Predictor, input, regressor, Covariate
print("The Features of the Iris Dataset is :\n",iris.feature_names ,"\n")
print("Target of the Iris Dataset is :\n", iris.target, "\n")
print("Target Names of the Iris Dataset is :\n", iris.target_names, "\n")
#Predicted values are known as Response, target, outcome, label and independent variable

print("Shape of Iris Data Set attribute is :\n", iris.data.shape,"\n")
print("Shape of Iris Data Set target is :\n", iris.target.shape,"\n")

X = iris.data
y= iris.target
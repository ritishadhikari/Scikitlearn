from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import binarize #to return specific limit values as desired
import pandas as pd
from sklearn.model_selection import cross_val_score

column_names = ['Pregnancies','Glucose','BP','Skin','Insulin','BMI','Pedigree','Age','Outcome']
pima = pd.read_csv(r"C:\Users\ritis\PycharmProjects\CSV Files\pima-indians-diabetes.csv",header=None, names=column_names)
print("Pima Dataframe :\n",pima.head(),"\n")
#print(pima.columns)
feature_cols = ['Pregnancies','Insulin','BMI','Age']
X= pima[feature_cols]
y = pima['Outcome']
X_Train, X_Test, y_Train, y_Test = train_test_split(X,y,random_state=1)

Lgg = LogisticRegression()
Lgg.fit(X_Train,y_Train)
y_pred_class = Lgg.predict(X_Test)
#print(y_pred_class)

print("Accuracy : ",accuracy_score(y_Test,y_pred_class),"\n")

print("Y Test Value Count :\n",y_Test.value_counts(),"\n")
print("Percentage of 1 :",y_Test[y_Test==1].count()/y_Test.count(),"\n")
print("Percentage of 0 :",y_Test[y_Test==0].count()/y_Test.count(),"\n")

print("Null Accuracy :", max(y_Test[y_Test==1].count()/y_Test.count(),y_Test[y_Test==0].count()/y_Test.count()),"\n")

print("True : ",y_Test.values[0:25])
print("Pred : ",y_pred_class[0:25],"\n")

confusion = confusion_matrix(y_Test,y_pred_class)
confusion_df = pd.DataFrame(data=confusion,index=['Actual[0]','Actual[1]'],columns=['Predicted[0]','Predicted[1]'])
print("The Confusion Matrix DataFrame Looks Like :\n", confusion_df,"\n")
print("Confusion Matrix : \n ",confusion, '\n')
TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]
#print(TP,TN,FP,FN)
print("Accuracy can also be found by :", (TP+TN)/sum(confusion.ravel()),"\n")
print("Classification Error can be found by : ", (FP+FN)/sum(confusion.ravel()),"\n")
print("Sensitivity - When the actual value is positive, how often is the prediction positive :", TP/(TP+FN),"\n")
print("Specificity - When the actual value is negative, how often is the prediction negative :", TN/(TN+FP),"\n")
print("False Positives - When the actual value is negative, how often is the prediction incorrect :", FP/(FP+TN),"\n")
print("Precision - When a Positive value is predicted, how often is it actually positive :", TP/(TP+FP),"\n")

#print(X_Test.values[0:10])
print("Printing the First 25 predicted response : ",Lgg.predict(X_Test)[0:25],"\n")
print("Printing the First 25 predicted probabilities : \n ",Lgg.predict_proba(X_Test)[0:25],"\n")

y_pred_prob = Lgg.predict_proba(X_Test)[:,1]
#print(y_pred_prob)

plt.subplot(2,2,1)
plt.hist(y_pred_prob, bins=8, rwidth=0.96)
plt.xlim(0,1)
plt.title("Histogram of predicted Probabilities")
plt.xlabel("Predicted Probability of Diabetes")
plt.ylabel("Frequency")


y_pred_class_later = binarize([y_pred_prob],0.35)[0]
print ("The first 10 values of y_pred_prob :",y_pred_prob[0:10], "\n")
print ("The first 10 predicted value : ",y_pred_class_later[0:10], "\n")
print("Accuracy : ",accuracy_score(y_Test,y_pred_class_later),"\n")
confusion = confusion_matrix(y_Test,y_pred_class_later)
confusion_df = pd.DataFrame(data=confusion,index=['Actual[0]','Actual[1]'],columns=['Predicted[0]','Predicted[1]'])
print("The Updated Confusion Matrix DataFrame Looks Like :\n", confusion_df,"\n")
print("Updated Confusion Matrix : \n ",confusion, '\n')
TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]
print("New Accuracy can also be found by :", (TP+TN)/sum(confusion.ravel()),"\n")
print("New Classification Error can be found by : ", (FP+FN)/sum(confusion.ravel()),"\n")
print("New Sensitivity - When the actual value is positive, how often is the prediction positive :", TP/(TP+FN),"\n")
print("New Specificity - When the actual value is negative, how often is the prediction negative :", TN/(TN+FP),"\n")
print("New False Positives - When the actual value is negative, how often is the prediction incorrect :", FP/(FP+TN),"\n")
print("New Precision - When a Positive value is predicted, how often is it actually positive :", TP/(TP+FP),"\n")

#For the ROC Curve
fpr,tpr,thresholds = roc_curve(y_Test,y_pred_prob)
plt.subplot(2,2,2)
plt.plot(fpr,tpr)
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("False Positive Rate (1-Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve")


def evaluate_threshold(threshold):
    print("sensitivity :", tpr[thresholds >threshold][-1],"\n")
    print("specificity :", 1- fpr[thresholds > threshold][-1],"\n")

evaluate_threshold(0.35)

#For AUC Score
auc = roc_auc_score(y_Test,y_pred_prob)
print("AUC Curve Score (higher the better) is :", auc, "\n")

#Finding the ROC AUC through Cross Val Score
score = cross_val_score(estimator=Lgg,X=X,y=y,scoring='roc_auc',cv=10).mean()
print("Roc Auc Score for the model through Cross Val Score is :\n", score, "\n")

plt.show()

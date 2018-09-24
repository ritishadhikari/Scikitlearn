import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer #Convert Text into Machine of Token Count
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

simple_train = ['call you tonight', 'call me a cab', 'please call me...please']
vect = CountVectorizer()
print("Vect Properties are: \n", vect, "\n")
print("Vect Output Properties are: \n", dir(vect), "\n")
vect.fit(simple_train)
print("Examining the Fitted Vocabulary :\n", vect.get_feature_names(),"\n")

simple_train_dtm_parse = vect.transform(simple_train)
print("The Type of the Parse Matrix is :\n", type(simple_train_dtm_parse),"\n")
print("The Document Term Sparse Matrix is :\n", simple_train_dtm_parse, "\n")

simple_train_dtm_dense = simple_train_dtm_parse.toarray()
print("The Type of the Dense Matrix is :\n", type(simple_train_dtm_dense),"\n")
print("The Document Term Dense Matrix is :\n", simple_train_dtm_dense, "\n")
print("Printing the Dense Matrix in a DataFrame \n", pd.DataFrame(data=simple_train_dtm_dense,index=simple_train,columns=vect.get_feature_names()), "\n")

simple_test = ['please don\'t call me']
simple_test_dtm = vect.transform(simple_test)
print("The Sparse Matrix for the Test Data Set is :\n", simple_test_dtm, "\n")
print("The Dense Matrix for the Test Data Set is :\n", simple_test_dtm.toarray(), "\n")
print("Printing the Dense Matrix in a DataFrame for the Test Data \n", pd.DataFrame(data=simple_test_dtm.toarray(),index=simple_test,columns=vect.get_feature_names()), "\n")


url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
sms = pd.read_table(filepath_or_buffer=url,header=None, names=['label','message'])
#file = r'C:\Users\ritis\PycharmProjects\CSV Files\sms.tsv'
#sms = pd.read_table(filepath_or_buffer=r'C:\Users\ritis\PycharmProjects\CSV Files\sms.tsv',sep='    ',header=None,names=['label','message'])
print("The Data Frame of sms looks like :\n", sms.head(10),"\n")
print("The Shape of the DataFrame is :\n", sms.shape,"\n")

X = sms['message']
print("The shape of the Input Message is :\n", X.shape, "\n")
y= sms['label']

print("The shape of the output Variable is \n", y.shape, "\n")


Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,random_state=1)
print("The Shape of the Training Input Data is :\n", Xtrain.shape)
print("The Shape of the Testing Input Data is :\n", Xtest.shape)
print("The Shape of the Training Output Data is :\n", ytrain.shape)
print("The Shape of the Testing Output Data is :\n", ytest.shape)

vect = CountVectorizer()

Xtrain_dtm_parse = vect.fit_transform(Xtrain)
print("The Sparse Matrix for the Training Data is :\n", Xtrain_dtm_parse, "\n")
print("The Dense Matrix for the Training Data is :\n", Xtrain_dtm_parse.toarray(), "\n")

fitted_vocab = vect.get_feature_names()
print("The Fitted Vocabulory is :\n", fitted_vocab, "\n")
print("The DataFrame Matrix for the Dense Matrix Train Data is :\n", pd.DataFrame(data=Xtrain_dtm_parse.toarray(),index =Xtrain ,columns=fitted_vocab))
#k = pd.DataFrame(data=Xtrain_dtm_parse.toarray(),index =Xtrain ,columns=fitted_vocab)
#k.to_excel(r'C:\Users\ritis\PycharmProjects\CSV Files\Dense Matrix.xlsx')

Xtest_dtm_parse = vect.transform(Xtest)
print("The Sparse Matrix for the Testing Data is :\n", Xtest_dtm_parse, "\n")
print("The Dense Matrix for the Training Data is :\n", Xtest_dtm_parse.toarray(), "\n")
print("The DataFrame Matrix for the Dense Matrix Test Data is :\n", pd.DataFrame(data=Xtest_dtm_parse.toarray(),index =Xtest ,columns=fitted_vocab), "\n")

mnb =MultinomialNB()
mnb.fit(Xtrain_dtm_parse,ytrain)
ypred =mnb.predict(Xtest_dtm_parse)
print('The Prediction are :\n', ypred,"\n")
accuracy = accuracy_score(ytest,ypred)
print("The Accuracy of the Multinomail Bayes Model is :\n", accuracy, "\n")


confusion =confusion_matrix(ytest,ypred)
print("The Confusion Metrics is :\n", pd.DataFrame(data=confusion,index=['ham','spam'],columns=['Negative','Positive']), "\n")

print ("Message Text for False Positives :\n", Xtest[ypred>ytest.values],"\n")
print ("Message Text for False Negative :\n", Xtest[ytest.values>ypred], "\n")
print("Message Text for True Positive \n", Xtest[(ytest=='spam') & (ypred=='spam')],"\n")
print("Message Text for False Negatives \n", Xtest[(ytest=='ham') & (ypred=='ham')],"\n")

ypred_prob = mnb.predict_proba(Xtest_dtm_parse)
print("The Probability Distribution is :\n", ypred_prob,"\n")
ypred_prob_positive =mnb.predict_proba(Xtest_dtm_parse)[:,1]
print("The Probability Distribution for the Positive Response are:\n", ypred_prob_positive,"\n")

'''auc_score = roc_auc_score(np.array(ytest),ypred_prob_positive)
print("AUC Score for Mnb :\n", auc_score, "\n")'''

print("Printing the First 50 fitted Vocabularies , Tokens \n", vect.get_feature_names()[0:50],"\n")
print("Printing the Last 50 fitted Vocabularies , Tokens \n", vect.get_feature_names()[-50:],"\n")
ham_spam_dist = pd.DataFrame(data=mnb.feature_count_,index=['Ham','Spam'],columns=vect.get_feature_names())
print("Ham and Spam distribution is :\n",ham_spam_dist ,"\n")
ham_dist = mnb.feature_count_[0,:]
print("Ham distribution is :\n", pd.DataFrame(data=ham_dist,index=vect.get_feature_names(),columns=['Ham']),"\n")
spam_dist = mnb.feature_count_[1,:]
print("Spam distribution is :\n", pd.DataFrame(data=spam_dist,index=vect.get_feature_names(),columns=['Spam']),"\n")

ham_spam_dist_transpose = ham_spam_dist.T
print("The Random Dataset and the Response count are :\n",ham_spam_dist_transpose.sample(n=5,random_state=6),"\n")

print("Ham and Spam overall counts are : \n", pd.DataFrame(data=mnb.class_count_,index=['Ham', 'Spam']),"\n")

ham_spam_dist_transpose['Ham'] = ham_spam_dist_transpose['Ham']+1
ham_spam_dist_transpose['Spam'] = ham_spam_dist_transpose['Spam']+1

print("The New Random Dataset and the Response count are :\n",ham_spam_dist_transpose.sample(n=5,random_state=6),"\n")

ham_spam_dist_transpose['Ham'] = ham_spam_dist_transpose['Ham']/mnb.class_count_[0]
ham_spam_dist_transpose['Spam'] = ham_spam_dist_transpose['Spam']/mnb.class_count_[1]
print("The Ham Spam distribution from the random dataset is now :\n", ham_spam_dist_transpose.sample(n=5,random_state=6), "\n")

ham_spam_dist_transpose['Spam Ratio'] = ham_spam_dist_transpose['Spam']/ham_spam_dist_transpose['Ham']
print("The Sorted Spam Ratio is :\n", ham_spam_dist_transpose.sort_values(by='Spam Ratio',ascending=False),"\n")

print("Checking for the Spam Ratio for a Particular Word :\n", ham_spam_dist_transpose.loc['dating','Spam Ratio'],'\n')



Xsample1 = '''You have won a claim worth prize 400 dollars and in this regards you are requested to send a deposit money of $ 500 and on the
reciept of the money, you will be awarded a guaranteed worth of prize money'''

Xsample2 = '''You are a good example to all the students in this class. I am very happy for your achievement and I wish you
all the success in your future endeavor'''

Xsampledf =pd.Series(Xsample1)
Xsample_dtm_parse = vect.transform(Xsampledf)
print("The Predicted Keyword is :\n",mnb.predict(Xsample_dtm_parse), "\n")


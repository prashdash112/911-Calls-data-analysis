import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix

loans = pd.read_csv('loan_data.csv')
X = loans.drop('not.fully.paid',axis=1)
y = loans['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
X_train = X_train.select_dtypes('float' or 'int')
X_test = X_test.select_dtypes('float' or 'int')

def Dtree(xtrain,ytrain,xtest,criterion='gini',max_depth=None):
    
    '''
    xtrain - Input features of train data 
    ytrain - Target features of train data 
    xtest -  Input features of test data
    '''
    
    model = DecisionTreeClassifier(criterion=criterion,max_depth=max_depth)
    model.fit(xtrain,ytrain)
    pred = model.predict(xtest)
    return pred 

if __name__ =='__main__':
    for i in range(1,20,2):
        result=Dtree(X_train,y_train,X_test,max_depth=i)
        print('---------------------------------------------------------\n','Value of max_depth:',i,'\n---------------------------------------------------------\n',classification_report(y_test,result))    

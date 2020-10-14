import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

advt=pd.read_csv('advertising.csv')
advt.drop(labels=['Ad Topic Line','City','Timestamp','Country'],inplace=True,axis=1)

X=advt[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y=advt['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=101)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

class logit:
    pred=[]
    
    
    def __init__(self,trainIP,trainOP,maxiter,testIP,testOP,solver='lbfgs'):
        self.trainIP=trainIP
        self.trainOP=trainOP
        self.maxiter=maxiter
        self.testIP=testIP
        self.testOP=testOP
        self.solver=solver
        
    def modelpipe(self):
        k=LogisticRegression(max_iter=self.maxiter,solver=self.solver)
        k.fit(self.trainIP,self.trainOP)
        logit.pred=k.predict(self.testIP)
        return logit.pred
    
    def report(self):
        return classification_report(self.testOP,logit.pred)
    
    
l1=  logit(X_train,y_train,200,X_test,y_test,solver='lbfgs')  

if __name__=='__main__':
    print('Predicted vals are: \n', l1.modelpipe())
    print('\n')
    print('Classification report is: \n',l1.report())
    

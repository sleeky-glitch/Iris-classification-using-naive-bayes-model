#coded by nishant singh tomar GitHub: sleeky_glitch
#imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
from sklearn.metrics import confusion_matrix
print("imports completed\n")

#reading and formating data
print("reading data\n")
data = pd.read_csv("datasets_19_420_Iris.csv")
data.drop(["Id"],axis=1,inplace=True)

#splitting data
print("splitting data into test and training set\n")
data.to_numpy() #converting data to numpy array
X = data.iloc[:,:4].values
y = data['Species'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 34)

# homeginizing data using fit and transform method
print("homeginizing data\n")
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#fitting and predicting data
print("training model \n")
classifier.fit(X_train,y_train)
y_predict=classifier.predict(X_test)
compare = np.vstack((y_predict,y_test)).T
print("comparing predicted output with test output \n",compare[:,:10],"\n")

#using confusion matrix to check validity of classifier
cm = confusion_matrix(y_predict,y_test)
print("confusion matirx is \n",cm,"\n")
total=cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2]
accuracy=(cm[1,1]+cm[2,2])/total
print("accuracy of classification model is ",accuracy )
input("press enter to continue\n")

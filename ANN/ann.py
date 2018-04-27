# Part 1: Data preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Part 2: Make ANN

#  Import Keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialize of the ANN
classifier = Sequential()

# Add Input Layers and First Hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu',input_dim=11 ))

# Add 2nd hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))    

# Add output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# Compiling ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'] )

# Fitting the ANN to training set
classifier.fit(X_train, y_train, batch_size= 10 , nb_epoch= 100 )

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting for a single customer
"""
Based on the model, predict whether the following customer will leave the bank
Geography - france
Credit Score - 600
Gender Male
Age 40
Tenure 3
Balance 60000
Number of Products 2
Has Credit Card Yes
Is Active Yes
Estimated Salary 50000
"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0,0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Evaluating the ANN using K-fold technique
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu',input_dim=11 ))
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy'] )
    return classifier
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=100 )
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs= -1)


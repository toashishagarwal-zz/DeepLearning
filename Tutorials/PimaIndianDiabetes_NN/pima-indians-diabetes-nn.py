##############################################################################
# Problem Statement: Given the information about Pima Indians as a csv, 
# predict whether the patient is diabetic or not
##############################################################################

# -------------------------------- 
#   PART 1: Data Preprocessing
# --------------------------------

# Import libraries
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

# provide a random seed to ensure reproducibility
np.random.seed(7)

# Load the dataset
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')

# Split the dataset into X (input variables) and Y (output variable)
X = dataset[:, 0:8]
Y = dataset[:,8]

# --------------------------------
#   PART 2: Make the ANN
# --------------------------------

# Create the model
model = Sequential()

# Add the input and hidden layers
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))

# Add the output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, Y, epochs=200, batch_size=10)

# Evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))





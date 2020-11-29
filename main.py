import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Datasets/ProcessedData.csv') #Take in our scaled, processed data

X = df.loc[:, df.columns != 'gentrif']
#Our input data is every column but the gentrified indicator column
y = df.loc[:, 'gentrif']
#Our output data is the gentrified indicator column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

y_train_list = y_train.values.tolist() #List our training set outcomes
weight_gentrif_by_this = (len(y_train_list)-sum(y_train_list))/(sum(y_train_list))
#This uses the unique characteristic of our y_train_list set, that it is composed
#of 0s and 1s, to find out what the ratio of '0' outcomes to '1' outcomes is.
gentrif_weights = {0:1.0, 1:weight_gentrif_by_this}
#This will assign outcomes some weight! We set the '0' outcome to have weight=1
#and the '1' outcome is much more rare, so it must be weighted proportionally to
#how rarely it shows up in our training set, so we set its weight to be higher!

model = Sequential()
#Create a simple neural network architecture which allows us to stack layers
model.add(Dense(14, activation='relu', input_dim=14))
#Input layer has 14 nodes for 14 inputs, a fairly standard node number, however
#we could immediately increase the node count here to give our model time to pick
#many potentially important features, or we could immediately decrease the node
#count here to force our model to quickly choose only a few variables or features
#that it thinks are most important, cutting down on accuracy somewhat, but also
#greatly reducing training time
model.add(Dense(7, activation='relu'))
#I used two hidden layers in this model, as using two layers gives the model
#enough complexity to classify a binary situation like this, even when the
#classification boundary has an arbitrarily complicated shape

#Furthermore, we use the 'relu' activation function, which only considers positive
#portions of the data passed to it, and while this may seem overly simple to use
#as an activation function, in conjunction with the weights and biases that are
#already being computed during the training process, this function is known to be
#incredibly effective, and its simplicity greatly decreases training time
model.add(Dense(1, activation='sigmoid'))
#On the output layer, use a sigmoid activation function to squash our predictions
#between 0 and 1, the only two outcomes we care about for a binary classification
#problem. We use a node value of 1, as our output should only have 1 dimension,
#as it should be a single number, either 0 or 1
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#Put the model together. The 'adam' optimizer adds on to the gradient descent
#characteristic of a normal neural network model. This gradient descent pushes
#the prediction error to go lower and lower, but usually we would have to fine-
#tune parameters which require very deep knowledge of the gradient descent process
#during the actual training. The 'adam' optimizer is useful because it requires
#very little parameter tuning, so we can build our model with greater ease!

#Use the'binary_crossentropy' loss function. This will be used to see how far
#our predictions are from our actual outcomes, using this binary_crossentropy
#loss function will tell the neural network we're looking at a binary clasification
#problem, and the loss function will be optimized for this situation

#The metrics parameter is fairly standard here, just to see how accurate our model
#is as we're training
model.fit(x=X_train,y=y_train, epochs=10, class_weight = gentrif_weights)
#Train the model! Input our training data into the model 10 times, while tweaking
#this program, I saw that after 10 epochs, the accuracy plateaud and we seemed to
#only be overfitting the model

#We give the model class weights, this is explained a bit below as well, but
#this is to make sure that the model considers both the gentrified and non-
#gentrified outcomes on equal grounds, despite the fact that the majority of
#our data is non-gentrified

scores = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: %.2f%%" % (scores[1]*100)) #Accuracy on training data
scores = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy: %.2f%%" % (scores[1]*100)) #Accuracy on testing data
#This matters much more, seeing how our model works on new data it hasn't seen yet

y_test_predictions = model.predict_classes(X_test)
y_test_binary = [bool(i) for i in y_test]

c_matrix = confusion_matrix(y_test_binary, y_test_predictions)
ax = sns.heatmap(c_matrix, annot=True, xticklabels=['Not Gentrified','Gentrified'],
    yticklabels=['Not Gentrified', 'Gentrified'], cbar=False, cmap='Reds')
#Putting together a confusion matrix displaying counts of predictions that were
#classified correctly, and predictions that were classified incorrectly
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
plt.show()
plt.clf()
#After looking at this matrix using fairly standard training procedures, it was
#clear that the imbalance of gentrified non-gentrified data was a problem. As
#long as our model predicted most of the data to be non-gentrified, it was likely
#right, as most of our data was non-gentrified. This was especially clear in the
#confusion matrix, as we saw that more often than not, actual gentrified data was
#predicted to be non-gentrified. This couldn't be seen on the accuracy measures of
#the model, as the vast majority of the data is non-gentrified and drowned this
#effect out. We fixed this above by weighting the gentrified data, giving it extra
#consideration in the model.

#Following this fix, the model categorizes more of the actual non-gentrified data
#wrongly, but most importantly, it now categorizes gentrified data as gentrified,
#much more often than not. The overall accuracy of the model suffers with this
#change, but this is good, as we want the model to be able to predict these
#minority cases of gentrified areas rather than just the majority non-gentrified
#data.

y_test_pred_probs = model.predict(X_test) #predicted classifications of inputs
FPR, TPR, _ = roc_curve(y_test, y_test_pred_probs) #comparing true vs predictions
plt.plot(FPR, TPR)
plt.plot([0,1],[0,1],'--', color='black') #diagonal line showing the worst case
plt.title('ROC Curve')
plt.xlabel('False Pos Rate')
plt.ylabel('True Pos Rate')
plt.show()
plt.clf()
#This curve is useful because we can look at the area between the worst case
#diagonal line on the plot and the curve to see how well our model can distinguish
#between gentrified and non-gentrified outcomes. The greater this area, the
#better. Our curve has a decent amount of area between itsself and the diagonal
#worst case line, showing that our model is fairly good at predicting between
#gentrified and non-gentrified data

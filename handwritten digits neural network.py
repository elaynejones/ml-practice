import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

#load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#get size of images
x_train.shape 

#scale the dataset - this will improve accuracy
#before, the values were from 0=255, now makes them all bn 0-1
x_train=x_train / 255
x_test=x_test / 255

#plot 1st train image
plt.matshow(x_train[0])

#show y value for that value
y_train[0]

#flatten train dataset - convert 28x28 image into one dimensional array
x_train_flat = x_train.reshape(len(x_train), 28*28)

#flatten test
x_test_flat = x_test.reshape(len(x_test), 28*28)

#make simple NN
#sequential - stack of layers, each layer is an element
#output shape is 10 - # of neurons
#input shape is the number of elements (784)
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])


model.compile(
    optimizer='adam', 
    loss = 'sparse_categorical_crossentropy', 
    metrics=['accuracy']
)


model.fit(x_train_flat, y_train, epochs=5)


model.evaluate(x_test_flat, y_test)

#doing some sample predictions
plt.matshow(x_test[0])

#this gives you the 10 neuron values
y_pred = model.predict(x_test_flat)
y_pred[0]

#find index of max score - this is the ultimate prediction
np.argmax(y_pred[0])

#find this label for all the values in y pred - so we can make cm later
y_pred_labels = [np.argmax(i) for i in y_pred]

#look at overall performance
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred_labels)

#visualize cm
import seaborn as sns 
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt ='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


#model still has lots of errors
#adding hidden layers to improve performance
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])


model.compile(
    optimizer='adam', 
    loss = 'sparse_categorical_crossentropy', 
    metrics=['accuracy']
)


model.fit(x_train_flat, y_train, epochs=5)

#evaluate 
model.evaluate(x_test_flat, y_test)

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred_labels)

#visualize new performance
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt ='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

#you can actually flatten built in w Keras API
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28))
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])


model.compile(
    optimizer='adam', 
    loss = 'sparse_categorical_crossentropy', 
    metrics=['accuracy']
)


model.fit(x_train, y_train, epochs=5)
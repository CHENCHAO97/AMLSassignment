# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 15:28:27 2019

@author: dell
"""
# train the hair_color classifier

###Part 1. import data###
import itertools


import pandas as pd
import numpy as np
from PIL import Image
import os, os.path
from sklearn.model_selection import train_test_split, GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# All_Images= pd.DataFrame(index = INDEX, columns = ['noisy','clean'])

All_Images = np.zeros((5000, 256, 256, 1))
X_total = np.zeros((4454, 256, 256, 1))

valid_images = ['.png', '.PNG']
for imagename in os.listdir('/Users/lichenchao/Desktop/AMLSassignment/dataset_face1/dataset_face'):
    # print(imagename)
    ind = int(str(os.path.splitext(imagename)[0]))
    # print(index)
    ext = os.path.splitext(imagename)[1]
    # print(ext)
    if ext.lower() not in valid_images:
        continue
    # All_Images[ind-1] = imageio.imread(os.path.join('dataset_face',imagename))
    img = Image.open(os.path.join('dataset_face', imagename)).convert('L')
    All_Images[ind] = np.array(img.getdata(), dtype=np.uint8).reshape(256, 256, 1)

count = 0
for i in range(5000):
    if sum(sum(All_Images[i])) != 0:
        # print(i)
        X_total[count] = All_Images[i]
        count = count + 1

labels = pd.read_csv('attribute_list_face.csv')

Y_total = np.array(labels['hair_color'])
# Convert the label class into a one-hot representation

num_classes = 6

row_delete = []
for i in range(4454):
    if Y_total[i] == -1:
        row_delete.append(i)

X_final = np.delete(X_total, row_delete, axis=0)
Y_final = np.delete(Y_total, row_delete, axis=0)

x_train, x_test, y_train, y_test = train_test_split(X_final, Y_final, test_size=0.2,shuffle=False)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def visualize_training(hist):
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['training', 'validation'], loc='lower right')
    plt.show()
    
    # A chart showing our training vs validation loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()




training_images = x_train / 255.0
training_labels = y_train
test_images = x_test/255.0
test_labels = y_test

model = keras.models.Sequential()

# add model layers
# model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(256, 256, 1)))
# model.add(Conv2D(16, kernel_size=3, activation='relu'))
model.add(keras.layers.Flatten())
# model.add(MaxPooling2D(pool_size = (2, 2), strides=2))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(6, activation='softmax'))

sgd = keras.optimizers.SGD(lr=0.0001, decay=0.0000006, momentum=0.9, nesterov=False)

model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


history=model.fit(training_images, training_labels, batch_size=128,validation_data=(test_images,test_labels),epochs=50)


res = model.evaluate(x=test_images, y=test_labels, batch_size=128, verbose=1, sample_weight=None, steps=None)
predictions = model.predict(test_images, batch_size=128, verbose=0, steps=None)
print('The classification accuracy on the test set is:', res[1])
print predictions
y_pred = np.argmax(predictions, axis=1)
test_labelsn= np.argmax(test_labels, axis=1)
print np.argmax(test_labels, axis=1)
print y_pred
print(confusion_matrix(test_labelsn, y_pred))
print(classification_report(test_labelsn, y_pred))

visualize_training(history)


cnf_matrix = confusion_matrix(test_labelsn, y_pred)
np.set_printoptions(precision=2)
class_names= ['bald','blond','ginger','brown','black','grey',]
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                  title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                  title='Normalized confusion matrix')

plt.show()











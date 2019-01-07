from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import landmarks as l2
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import tensorflow as tf
from tensorflow import keras
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    
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
    
    
def get_data():
    X, y = l2.extract_features_labels()
    #print len(X)
    Y = np.array([y, -(y - 1)]).T
    #print len(Y)
    tr_X = X[:3000]
    tr_Y = Y[:3000]
    te_X = X[3000:]
    te_Y = Y[3000:]
    return tr_X, tr_Y, te_X, te_Y

tr_X, tr_Y, te_X, te_Y = get_data()

training_images=tr_X
training_labels=tr_Y
test_images=te_X
test_labels=te_Y

sizeTr=len(training_images)
sizeTe=len(test_images)
TD_train = training_images.reshape(sizeTr,-1)
TD_test = test_images.reshape(sizeTe,-1)

svclassifier = svm.SVC(kernel='linear',class_weight='balanced')
svclassifier.fit(TD_train, training_labels[:,0])
y_pred = svclassifier.predict(TD_test)
print(y_pred)
#print(training_labels)
print(confusion_matrix(test_labels[:,0], y_pred))
print(classification_report(test_labels[:,0], y_pred))


#plot confusion matrix
cnf_matrix = confusion_matrix(test_labels[:,0], y_pred)
np.set_printoptions(precision=2)
class_names= ['old','young']
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                  title='Confusion matrix, without normalization')


plt.show()




















import joblib
import keyVariables as kv
import trainTestSplitting as tts
import preProcessing as pp
import matplotlib.pyplot as plt
import numpy as np

model=joblib.load("movelytics84.75.pkl")

def predict(path):
    predictVideo=pp.preprocessor(path)
    allLabels=model.predict(np.expand_dims(predictVideo,axis=0))[0]
    predictedIndex=np.argmax(allLabels)
    predictedClass=kv.CLASS_LABLES[predictedIndex]
    return predictedClass

def summary():
    model.summary()

def accuracy():
    a,b,featureTest,c,d,labelTrain=tts.trainTestSplit()
    model.evaluate(featureTest,labelTrain)

def graph():
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(model.history['accuracy'])
    plt.plot(model.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.subplot(1, 2, 2)
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()



    

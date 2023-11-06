import joblib
import preProcessing as pp
import numpy as np
import keyVariables as kv

model=joblib.load(filename="movelytics84.75.pkl")
def predict(path):
    predictVideo=pp.preprocessor(path)
    allLabels=model.predict(np.expand_dims(predictVideo,axis=0))[0]
    predictedIndex=np.argmax(allLabels)
    predictedClass=kv.CLASS_LABLES[predictedIndex]
    return predictedClass

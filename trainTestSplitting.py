import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import preProcessing as pp

def featureAndLabel():
    le=LabelEncoder()
    featureList,labelList=pp.Extractor()
    featureList=np.asarray(featureList,dtype=np.float32)
    labelList=le.fit_transform(np.array(labelList))
    labelList=to_categorical(labelList)
    return featureList,labelList

def trainTestSplit(testSize=0.1,validationSize=0.35):
    featureList,labelList=featureAndLabel()
    featureTemp,featureTest,labelTemp,labelTest=train_test_split(
    featureList,
    labelList,
    shuffle=True,
    test_size=testSize
    )

    featureTrain,featureValidate,labelTrain,labelValidate=train_test_split(
        featureTemp,
        labelTemp,
        shuffle=True,
        test_size=validationSize
    )

    return featureTrain,featureValidate,featureTest,labelTrain,labelValidate,labelTest

import keyVariables as kv
import cv2
import numpy as np

def preprocessor(path):
    frameList=[]
    currentFrame=0
    cam=cv2.VideoCapture(path)
    frameCount=int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    frameCut=max(1,int(frameCount/kv.SEQUENCE_LENGTH))
    while True:
        ret,frame=cam.read()
        if ret:
            if currentFrame%frameCut==0 and len(frameList)!=kv.SEQUENCE_LENGTH:
                resizedFrame=cv2.resize(frame,(kv.IMAGE_HEIGHT,kv.IMAGE_WIDTH))
                normalizedFrame=resizedFrame/255
                frameList.append(normalizedFrame)
            currentFrame+=1
        else:
            break
    cam.release()
    if len(frameList)==kv.SEQUENCE_LENGTH:
        return np.asarray(frameList,np.float32)

def Extractor():
    featureList=[]
    labelList=[]
    for i in kv.CLASS_LABLES:
        joinedPath=kv.PATH+"/"+i
        newPath=kv.listdir(joinedPath)
        for j in newPath:
            j=joinedPath+"/"+j
            frameList=preprocessor(j)
            if (len(frameList)==kv.SEQUENCE_LENGTH): 
                featureList.append(np.asarray(frameList,np.float32))
                labelList.append(i)      
    return featureList,labelList


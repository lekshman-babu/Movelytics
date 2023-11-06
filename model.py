import keyVariables as kv
import trainTestSplitting as tts
import joblib
from keras import Sequential
from keras.layers import MaxPooling2D, Dense, Flatten, TimeDistributed, Dropout, LSTM, Conv2D, BatchNormalization
from keras.callbacks import EarlyStopping

def baseModel():
    model=Sequential(name="Movelytics")
    model.add(TimeDistributed(Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(kv.SEQUENCE_LENGTH,kv.IMAGE_HEIGHT,kv.IMAGE_WIDTH,3))))
    model.add(TimeDistributed(BatchNormalization()))
    # model.add(TimeDistributed(Conv2D(32,(3,3),activation='relu',padding='same')))
    model.add(TimeDistributed(MaxPooling2D()))
    model.add(TimeDistributed(Dropout(0.3)))

    model.add(TimeDistributed(Conv2D(16,(3,3),activation='relu',padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    # model.add(TimeDistributed(Conv2D(8,(3,3),activation='relu',padding='same')))
    # model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D()))
    model.add(TimeDistributed(Dropout(0.6)))

    model.add(TimeDistributed(Conv2D(8,(5,5),activation='relu',padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Conv2D(8,(3,3),activation='relu',padding='same')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D()))
    model.add(TimeDistributed(Dropout(0.7)))

    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128,activation='tanh',recurrent_activation='sigmoid',return_sequences=True))
    model.add(LSTM(101,activation='tanh',recurrent_activation='sigmoid',return_sequences=False))

    model.add(Dense(len(kv.CLASS_LABLES),activation='softmax'))

    return model

def compileAndFit():
    model=baseModel()
    early_stopping = EarlyStopping(
        monitor='val_loss',   
        patience=5,           
        restore_best_weights=True,  
        mode='min',
        start_from_epoch=6
    )
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    featureTrain,featureValidate,featureTest,labelTrain,labelValidate,labelTest=tts.trainTestSplit()
    modelHist=model.fit(
    featureTrain,
    labelTrain,
    validation_data=(featureValidate,labelValidate),
    epochs=kv.EPOCHS,
    batch_size=kv.BATCH_SIZE,
    callbacks=[early_stopping]
    )
    return modelHist

def saveModel():
    joblib.dump(compileAndFit(),"modelFinal")
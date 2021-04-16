from sklearn import preprocessing
import numpy as np
import math
from keras.utils import to_categorical
import numpy as np
import csv, math, os, time, sys, pickle, psutil, itertools
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.optimizers import Adam, SGD
from keras.callbacks import TerminateOnNaN
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from keras import backend as K
from keras.utils import to_categorical


class LSTM_input:
  def __init__(self, timeseries):
    self.data_series = timeseries
    self.dataX = []
    self.dataY = []
    self.trainX = []
    self.trainY = []
    self.valX = []
    self.valY = []
    self.testX = []
    self.testY = []
    self.trainX_categor = []
    self.trainY_categor = []
    self.valX_categor = []
    self.valY_categor = []
    self.testX_categor = []
    self.testY_categor = []
    self.dataX_in = []
    self.dataY_in = []

  def timeseries_to_history_seq(self, history_length):
    dataX, dataY = [], []
    for i in range(len(self.data_series) - history_length):
      self.dataX.append(self.data_series[i: i + history_length])
      self.dataY.append(self.data_series[i + history_length])

  def split_data(self, ratio):
  
    samples = np.array(self.dataY).shape[0]
    test_samples = (ratio) * samples
    samples_interval = 1
    if test_samples != 0:
      samples_interval = math.floor(samples / test_samples)
    s = 0
    while s < samples:
      if s % samples_interval == 0:
        self.valX.append(self.dataX[s])
        self.valY.append(self.dataY[s])
        s += 1
        self.testX.append(self.dataX[s])
        self.testY.append(self.dataY[s])
      else:
        self.trainX.append(self.dataX[s])
        self.trainY.append(self.dataY[s])
      s += 1
  
  def to_categor(self, num_classes):
    # shape is [samples, time steps, features]
    inter = to_categorical(np.array(self.trainX), num_classes = num_classes)
    self.trainX_categor = np.reshape(inter, (inter.shape[0], inter.shape[1], num_classes))
    inter = to_categorical(np.array(self.valX), num_classes = num_classes)
    self.valX_categor = np.reshape(inter, (inter.shape[0], inter.shape[1], num_classes))
    inter = to_categorical(np.array(self.testX), num_classes = num_classes)
    self.testX_categor = np.reshape(inter, (inter.shape[0], inter.shape[1], num_classes))
    
    # shape is [samples, features]
    self.trainY_categor = to_categorical(self.trainY, num_classes=num_classes)
    self.valY_categor = to_categorical(self.valY, num_classes=num_classes)
    self.testY_categor = to_categorical(self.testY, num_classes=num_classes)
    
  def prepare(self):
    # shape is [samples, time steps, features]
    self.dataX = np.array(self.dataX)
    self.dataX_in = np.reshape(np.array(self.dataX), (self.dataX.shape[0], self.dataX.shape[1], 1))
  # shape is [samples, features]
    self.dataY = np.array(self.dataY)
    self.dataY_in = np.reshape(np.array(self.dataY), (self.dataY.shape[0], 1))
    


class LSTM_model:
  def __init__(self, input):
    self.input = input
    self.model = None

  def create(self, layers, learning_rate, dropout, history_length, num_classes):
    self.model = Sequential()
    self.model.add(LSTM(layers, input_shape=(history_length, num_classes), return_sequences=True, recurrent_dropout=dropout))
    self.model.add(LSTM(layers))
    self.model.add(Dense(num_classes, activation='softmax')) #not softmax
  
    # Optimizer, loss function, accuracy metrics
    self.model.compile(optimizer=SGD(lr=learning_rate), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    #self.model.compile(optimizer=Adam(lr=learning_rate), loss = 'mean_squared_error')

    print self.model.summary()
    
  def train(self):
    self.model.fit(self.input.trainX_categor, self.input.trainY_categor, epochs = 100, validation_data=(self.input.trainX_categor, self.input.trainY_categor))
    #self.model.fit(self.input.dataX_in, self.input.dataY_in, epochs = 100)
    self.model.save("lstm_page_539")
    
  def infer(self):
    predictY = self.model.predict(self.input.trainX_categor)
    #print self.input.dataY_in
    #print predictY
    
    dataY = np.array([np.argmax(x) for x in self.input.trainY_categor])
    predictY = np.array([np.argmax(x) for x in predictY])
    
    print dataY
    print predictY


  def calculate_prediction_error(self, dataY, predictY):
    print dataY.shape, predictY.shape
    err = 0.0
    dataY = np.array([np.argmax(x) for x in dataY])
    predictY = np.array([np.argmax(x) for x in predictY])
    for i in range(0, dataY.shape[0]):
      if dataY[i] != predictY[i]:
        err += 1
    error = (err / dataY.shape[0]) * 100
    return error
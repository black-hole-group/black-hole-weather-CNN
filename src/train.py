import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import scipy.misc

import datetime
import os
import numpy as np
from numpy import newaxis

from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model, load_model

#from data import get_data
from models import create_auto_encoder
from losses import r_squared, CustomLoss
from params import args
from params import write_results
#from CyclicLearningRate import CyclicLR

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


def main():
    X = np.load("/DL/dl_coding/DL_code/Data/x.npy")
    Y = np.load("/DL/dl_coding/DL_code/Data/y.npy")

    size_train = int(0.8*X.shape[0])
    size_val = int(0.1*X.shape[0])
    
    print(size_train)
    print(size_val)
    
    print(X[:size_train].shape)
    print(Y[:size_train].shape)
    #print(mass[:size_train].shape)
    model = create_auto_encoder(filters=args.filters)

    model = multi_gpu_model(model, gpus=2)

    folder_name = args.results_path
    os.makedirs(folder_name)
    trained_weights = folder_name + 'dl_fluids.h5'

    optimizer = Adam(0.0005)
    
    callbacks = [EarlyStopping(patience=20, verbose=1), ModelCheckpoint(trained_weights, verbose=1, save_best_only=True)]

    model.compile(optimizer=optimizer, loss={'prediction': CustomLoss(args.alpha, args.beta, args.gamma, args.delta)}, metrics=[r_squared])

    start_training = datetime.datetime.now()
    history = model.fit(X[:size_train], Y[:size_train], epochs=args.epochs, batch_size=args.batch_size, validation_data=(X[size_train:int(size_train+size_val)], Y[size_train:int(size_train+size_val)]), callbacks=callbacks)
    end_training = datetime.datetime.now()
    
    save_loss = open(folder_name + "loss_values.txt", "w")
    save_loss.write(str(history.history['loss']) + "\n")
    save_loss.write(str(history.history['val_loss']) + "\n")
    
    #scores = model.evaluate(X[int(size_train+size_val):], Y[int(size_train+size_val):], verbose=1)
 

if __name__ == '__main__':
    main()

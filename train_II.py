import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import scipy.misc

import datetime
import os
import numpy as np
from numpy import newaxis


import tensorflow as tf

from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model, load_model

from models import create_auto_encoder
from losses import r_squared, LossCustom
from params import args
from params import write_results


def generator(n_batch):

         X = np.load("/DL/dl_coding/DL_code/Data/x.npy")
         ix = np.random.randint(0, X.shape[0], 500)
         xt = X[ix]
         del X
         Y = np.load("/DL/dl_coding/DL_code/Data/y.npy")
         yt = Y[ix]
         del Y
         idx = np.random.randint(0, xt.shape[0], n_batch)
         while True:
           batch_X, batch_Y = xt[idx], yt[idx]
           yield batch_X, batch_Y

def main():


    model = create_auto_encoder(filters=args.filters)
    model = multi_gpu_model(model, gpus=2)
    optimizer = Adam(0.0005)
    model.compile(optimizer=optimizer, loss=LossCustom(args.alpha, args.beta), metrics=[r_squared])


    folder_name = args.results_path
    os.makedirs(folder_name)
    trained_weights = folder_name + 'dl_fluids.h5'

    callbacks = [EarlyStopping(patience=20, verbose=1), ModelCheckpoint(trained_weights, verbose=1, save_best_only=True)]

    start_training = datetime.datetime.now()
    model.fit_generator(generator(args.batch_size), steps_per_epoch=int(2670/args.batch_size), epochs=args.epochs,verbose=1, callbacks=callbacks, \
                        validation_data=generator(200), validation_steps=(int(200/args.batch_size)))
    end_training = datetime.datetime.now()

    print(end_training - start_training)
    #scores = model.evaluate(X_test, Y_test, verbose=1)
    write_results(folder_name, (end_training - start_training))


if __name__ == '__main__':
    main()
           

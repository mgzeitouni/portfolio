# -*- coding: utf-8 -*-
import gc
import datetime
import time
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import math
import random
import pickle
import pdb
import csv
import sys

from training_functions import *

def worker(sys):
    
    if len(sys.argv) <2:
        print("First arg must be a coin")
        sys.exit()
        
    coin_predict = sys.argv[1]
    
    if coin_predict not in ['BTC','ETH']:
        print("Invalid coin")
        sys.exit()
        
    import keras
    from keras.models import Sequential
    from keras.layers import Activation, Dense
    from keras.layers import LSTM
    from keras.layers import Dropout
        
    btc_data = get_market_data("bitcoin", tag='BTC')
    eth_data = get_market_data("ethereum", tag='ETH')
    
    # Merges 2 coin data into 1 df
    data = merge_data([btc_data,eth_data])
    
    # Drops unneccessary columns
    data = create_model_data(data)

    # Split into train and test
    samples = np.random.choice(len(data), int(math.floor(training_size * len(data))), replace=False)
    
    if sys.argv[2] == 'new_csv':
        with open("crypto_training_rmse_%s.csv" %coin_predict,"wb") as data_file:
            writer=csv.writer(data_file)
            writer.writerow(["neurons",	"layers",	"activation_function",	"optimizer",	"dropout",	"batch_size",	"epochs",	"input_window_len",	"output_window_len",	"keep_order",	"elapsed_time",	"loss(rmse)",	"val_loss(rmse)",	"mse",	"val_mse","mae","val_mae"])
            
    while(True):
        
        # try:
        
            loss = 'mse' 
            
            neurons, layers, activation_function,optimizer, dropout, batch_size, epochs, input_window_len, output_window_len, keep_order = choose_params()
            epochs = 25
            # neurons = 17
            # layers = 3
            activation_function = 'relu'
            # optimizer = 'adam'
            # dropout=0.4
            # batch_size=80
            # epochs = 10
            # input_window_len = 7
            # output_window_len = 2
            # keep_order = False
            print ("==============Start Model===============")
            print("Neurons: %s"%neurons)
            print("Layers: %s"%layers)
            print("Activation: %s"%activation_function)
            print("Optimizer: %s"%optimizer)
            print("Dropout: %s"%dropout)
            print("Batch Size: %s"%batch_size)
            print("Epochs: %s"%epochs)
            print("Input Window Len: %s"%input_window_len)
            print("Output Window Len: %s"%output_window_len)
            print("Keep Order: %s"%keep_order)
            
            X_train, X_test = create_inputs(data, keep_order, samples,input_window_len, output_window_len, prediction=False)
            X_train, X_test = to_array(X_train, prediction=False), to_array(X_test, prediction=False)
            
                
            Y_train, Y_test = create_outputs(data,samples,keep_order, coin_predict,input_window_len, output_window_len)
           # Y_train_btc, Y_test_btc = create_outputs(data,samples,keep_order, 'BTC',input_window_len, output_window_len)
           # Y_train_eth, Y_test_eth = create_outputs(data, samples, keep_order,'ETH',input_window_len, output_window_len)
    
              # clean up the memory
            gc.collect()
            # random seed for reproducibility
           # np.random.seed(202)
            # initialise model architecture
            btc_model = build_model(X_train, 1, neurons, activation_function, dropout, loss, optimizer, layers)
            earlystop = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=3,
                              verbose=0, mode='auto')
            callbacks_list = [earlystop]
            start = time.time()
            # train model on data
            btc_history = btc_model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, verbose=1,validation_data=(X_test, Y_test), shuffle=False)
            
            end = time.time()
            
            elasped_minutes = (end-start)/60
            
            formatted_time = '{0:.2f}'.format(elasped_minutes)
      
            model_loss = btc_history.history['loss'][0]
            val_loss=btc_history.history['val_loss'][0]
            mse = btc_history.history['mean_squared_error'][0]
            val_mse = btc_history.history['val_mean_squared_error'][0]
            mae = btc_history.history['mean_absolute_error'][0]
            val_mae = btc_history.history['val_mean_absolute_error'][0]
            
            row = [neurons, layers, activation_function,optimizer, dropout, batch_size, epochs, input_window_len, output_window_len, keep_order,formatted_time, model_loss, val_loss, mse, val_mse, mae, val_mae]
            
            with open("crypto_training_rmse_%s.csv"%coin_predict,"a") as data_file:
                writer=csv.writer(data_file)
                writer.writerow(row)
      
            print ("==============End Model===============")
        
           # predict(btc_model)
            # serialize model to JSON
            # model_json = btc_model.to_json()
            # with open("model.json", "w") as json_file:
            #     json_file.write(model_json)
            # # serialize weights to HDF5
            # btc_model.save_weights("model.h5")
            # print("Saved model to disk")

        # except:
        #     print("Error with model")
            
if __name__=="__main__":
    
   num_threads = int(sys.argv[3])

   threads = []
   for i in range(num_threads):
       t = threading.Thread(target=worker, args=(sys,))
       threads.append(t)
       t.start()
    
    #worker(sys)
    #X_predict = predict()
    

    

  
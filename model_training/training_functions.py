#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 00:14:58 2018

@author: morriszeitouni
"""
import pandas as pd
import numpy as np
import datetime
import time
import random
import threading
import pdb

merge_date = '2014-01-01'
training_size = 0.85
loss='mse'
#combos = len(neurons) * len(activation_functions) * len(optimizer) * len(dropout) * len(batch_size) * len(epochs) * len(input_window_len) * len(output_window_len) * len(keep_order)

def choose_params():
        
    neurons = [0, 750]   # 8     
    layers = [2,20]        
    activation_functions = ['relu','tanh'] # 3 
                   
    optimizer=['adam','sgd','rmsprop']    # 4
    dropout = [0,1]            # 6
    batch_size = [5,200]      # 7  
    epochs = [10,40]                 # 6
    input_window_len = [7,50]       # 6
    output_window_len = [2,10]    # 9 
    keep_order=[True,False]        # 2 
    
    n = random.randint(neurons[0],neurons[1])    
    l = random.randint(layers[0],layers[1])    
    a = random.choice(activation_functions)
    o = random.choice(optimizer)
    d = round(random.random(),2)
    b = random.randint(batch_size[0],batch_size[1])
    e = random.randint(epochs[0],epochs[1])
    i =  random.randint(input_window_len[0],input_window_len[1])
    out = random.randint(output_window_len[0],output_window_len[1])
    k = random.choice(keep_order)
            
    
        
    return n,l,a,o,d,b,e,i,out,k

def get_market_data(market, tag=True):
  """
  market: the full name of the cryptocurrency as spelled on coinmarketcap.com. eg.: 'bitcoin'
  tag: eg.: 'btc', if provided it will add a tag to the name of every column.
  returns: panda DataFrame
  This function will use the coinmarketcap.com url for provided coin/token page. 
  Reads the OHLCV and Market Cap.
  Converts the date format to be readable. 
  Makes sure that the data is consistant by converting non_numeric values to a number very close to 0.
  And finally tags each columns if provided.
  """
  market_data = pd.read_html("https://coinmarketcap.com/currencies/" + market + 
                             "/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"), flavor='html5lib')[0]

  market_data = market_data.assign(Date=pd.to_datetime(market_data['Date']))  
  market_data['Volume'] = (pd.to_numeric(market_data['Volume'], errors='coerce').fillna(0))
  if tag:
    market_data.columns = [market_data.columns[0]] + [tag + '_' + i for i in market_data.columns[1:]]
  return market_data

def merge_data(df_array, from_date=merge_date):
  """
  a: first DataFrame
  b: second DataFrame
  from_date: includes the data from the provided date and drops the any data before that date.
  returns merged data as Pandas DataFrame
  """
  # Merge first 2
  merged_data = pd.merge(df_array[0],df_array[1], on=['Date'])
  # Then loop through to keep merging
  for i in range(2,len(df_array)):
      merged_data = pd.merge(merged_data,df_array[i], on=['Date'])
      merged_data = merged_data[merged_data['Date'] >= from_date]
  return merged_data


def add_volatility(data, coins=['BTC', 'ETH']):
  """
  data: input data, pandas DataFrame
  coins: default is for 'btc and 'eth'. It could be changed as needed
  This function calculates the volatility and close_off_high of each given coin in 24 hours, 
  and adds the result as new columns to the DataFrame.
  Return: DataFrame with added columns
  """
  for coin in coins:
    # calculate the daily change
    kwargs = {coin + '_change': lambda x: (x[coin + '_Close'] - x[coin + '_Open']) / x[coin + '_Open'],
             coin + '_close_off_high': lambda x: 2*(x[coin + '_High'] - x[coin + '_Close']) / (x[coin + '_High'] - x[coin + '_Low']) - 1,
             coin + '_volatility': lambda x: (x[coin + '_High'] - x[coin + '_Low']) / (x[coin + '_Open'])}
    data = data.assign(**kwargs)
  return data


def create_model_data(data):
  """
  data: pandas DataFrame
  This function drops unnecessary columns and reverses the order of DataFrame based on decending dates.
  Return: pandas DataFrame
  """
  #data = data[['Date']+[coin+metric for coin in ['btc_', 'eth_'] for metric in ['Close','Volume','close_off_high','volatility']]]
  data = data[['Date']+[coin+metric for coin in ['BTC_', 'ETH_'] for metric in ['Close','Volume']]]
  data = data.sort_values(by='Date')
  data = data.drop('Date', 1)
  return data


def split_data(data, training_size=0.8):
  """
  data: Pandas Dataframe
  training_size: proportion of the data to be used for training
  This function splits the data into training_set and test_set based on the given training_size
  Return: train_set and test_set as pandas DataFrame
  """

  for number in np.arange(len(data)):
     
      if number in random_samples:
          train_set.loc[len(train_set)] = data.loc[number]
      else:
          test_set.loc[len(test_set)] = data.loc[number]
  
  #return data[:int(training_size*len(data))], data[int(training_size*len(data)):]
  return train_set, test_set


def create_inputs(data, keep_order, samples,input_window_len, output_window_len, coins=['BTC', 'ETH'], prediction =False):
  """
  data: pandas DataFrame, this could be either training_set or test_set
  coins: coin datas which will be used as the input. Default is 'btc', 'eth'
  window_len: is an intiger to be used as the look back window for creating a single input sample.
  This function will create input array X from the given dataset and will normalize 'Close' and 'Volume' between 0 and 1
  Return: X, the input for our model as a python list which later needs to be converted to numpy array.
  """
  norm_cols = [coin + metric for coin in coins for metric in ['_Close', '_Volume']]
  inputs = []
  k=0
  if prediction:
      k=1
      
  for i in range(len(data) - input_window_len+k):
    temp_set = data[i:(i + input_window_len)].copy()
    inputs.append(temp_set)
    for col in norm_cols:
      inputs[i].loc[:, col] = inputs[i].loc[:, col] / inputs[i].loc[:, col].iloc[0] - 1  
  
  all_inputs = inputs[:-output_window_len]
  train_set = []
  test_set = []
  
  if prediction:
    return inputs[-1]

  else:
    if keep_order:

        train_set = all_inputs[:(int(training_size*len(data)))]
        test_set = all_inputs[(int(training_size*len(data))):]
    else:
    
        for number in np.arange(len(all_inputs)):
          
            if number in samples:
                train_set.append( all_inputs[number])
            else:
                test_set.append(all_inputs[number])
    

    return train_set, test_set


def create_outputs(data, samples, keep_order, coin, input_window_len, output_window_len):
  """
  data: pandas DataFrame, this could be either training_set or test_set
  coin: the target coin in which we need to create the output labels for
  window_len: is an intiger to be used as the look back window for creating a single input sample.
  This function will create the labels array for our training and validation and normalize it between 0 and 1
  Return: Normalized numpy array for 'Close' prices of the given coin
  """

  outputs = (data[coin + '_Close'][input_window_len+output_window_len:].values / data[coin + '_Close'][:-(input_window_len+output_window_len)].values) - 1

  train_set = []
  test_set = []
  
  if keep_order:

      train_set = outputs[:(int(training_size*len(data)))]
      test_set = outputs[(int(training_size*len(data))):]
  else:
      
      for number in np.arange(len(outputs)):
         
          if number in samples:
              train_set.append( outputs[number])
          else:
              test_set.append(outputs[number])
          
  return train_set, test_set

def to_array(data, prediction):
  """
  data: DataFrame
  This function will convert list of inputs to a numpy array
  Return: numpy array
  """
  if prediction:
    x = [np.array(data)]
  else: 
    x = [np.array(data[i]) for i in range (len(data))]
  return np.array(x)

def show_plot(data, tag):
  fig, (ax1, ax2) = plt.subplots(2,1, gridspec_kw = {'height_ratios':[3, 1]})
  ax1.set_ylabel('Closing Price ($)',fontsize=12)
  ax2.set_ylabel('Volume ($ bn)',fontsize=12)
  ax2.set_yticks([int('%d000000000'%i) for i in range(10)])
  ax2.set_yticklabels(range(10))
  ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
  ax1.set_xticklabels('')
  ax2.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
  ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,7]])
  ax1.plot(data['Date'].astype(datetime.datetime),data[tag +'_Open'])
  ax2.bar(data['Date'].astype(datetime.datetime).values, data[tag +'_Volume'].values)
  fig.tight_layout()
  plt.show()
  

def date_labels():
  last_date = market_data.iloc[0, 0]
  date_list = [last_date - datetime.timedelta(days=x) for x in range(len(X_test))]
  return[date.strftime('%m/%d/%Y') for date in date_list][::-1]


def plot_results(history, model, Y_target, coin):
  plt.figure(figsize=(25, 20))
  plt.subplot(311)
  plt.plot(history.epoch, history.history['loss'], )
  plt.plot(history.epoch, history.history['val_loss'])
  plt.xlabel('Number of Epochs')
  plt.ylabel('Loss')
  plt.title(coin + ' Model Loss')
  plt.legend(['Training', 'Test'])

  plt.subplot(312)
  plt.plot(Y_target)
  plt.plot(model.predict(X_train))
  plt.xlabel('Dates')
  plt.ylabel('Price')
  plt.title(coin + ' Single Point Price Prediction on Training Set')
  plt.legend(['Actual','Predicted'])

  ax1 = plt.subplot(313)
  plt.plot(test_set[coin + '_Close'][window_len:].values.tolist())
  plt.plot(((np.transpose(model.predict(X_test)) + 1) * test_set[coin + '_Close'].values[:-window_len])[0])
  plt.xlabel('Dates')
  plt.ylabel('Price')
  plt.title(coin + ' Single Point Price Prediction on Test Set')
  plt.legend(['Actual','Predicted'])
  
  date_list = date_labels()
  ax1.set_xticks([x for x in range(len(date_list))])
  for label in ax1.set_xticklabels([date for date in date_list], rotation='vertical')[::2]:
    label.set_visible(False)

  plt.show()

def root_mean_squared_error(y_true, y_pred):
  from keras import backend as K
  return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def build_model(inputs, output_size, neurons, activation_function, dropout, loss, optimizer, layers):
  """
  inputs: input data as numpy array
  output_size: number of predictions per input sample
  neurons: number of neurons/ units in the LSTM layer
  active_func: Activation function to be used in LSTM layers and Dense layer
  dropout: dropout ration, default is 0.25
  loss: loss function for calculating the gradient
  optimizer: type of optimizer to backpropagate the gradient
  This function will build 3 layered RNN model with LSTM cells with dripouts after each LSTM layer 
  and finally a dense layer to produce the output using keras' sequential model.
  Return: Keras sequential model and model summary
  """
  import keras
  from keras.models import Sequential
  from keras.layers import Activation, Dense
  from keras.layers import LSTM
  from keras.layers import Dropout
    
  model = Sequential()
 
  model.add(LSTM(neurons, return_sequences=True, input_shape=(inputs.shape[1], inputs.shape[2]), activation=activation_function))
  model.add(Dropout(dropout))
  
  for layer in range(layers-1):
      model.add(LSTM(neurons, return_sequences=True, activation=activation_function))
      model.add(Dropout(dropout))
      
  model.add(LSTM(neurons, activation=activation_function))
  model.add(Dropout(dropout))
  model.add(Dense(units=output_size))
  model.add(Activation(activation_function))
  model.compile(loss=root_mean_squared_error, optimizer=optimizer, metrics=['mae', 'mse'])
  model.summary()
  return model

def predict(model):
  

    btc_data = get_market_data("bitcoin", tag='BTC')
    eth_data = get_market_data("ethereum", tag='ETH')
        # Merges 2 coin data into 1 df
    data = merge_data([btc_data,eth_data])

    # Drops unneccessary columns
    data = create_model_data(data)

    samples=[]
    
    X_predict= create_inputs(data, True, samples,13, 2,prediction=True )

    X_predict = to_array(X_predict, prediction=True)
   # pdb.set_trace()
    
    import keras

    if model:
       prediction = model.predict(X_predict)
   # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    
    prediction = loaded_model.predict(X_predict)
    print(prediction)
    return X_predict


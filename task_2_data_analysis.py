import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests
import json
import sys

def plot_metrics(history, string):
    """ Function to plot training and validation metrics per epoch

    Args:
        history (keras.engine.sequential.Sequential): keras' model.history
        string (str): metric name (accuracy, mse, mae, etc...)
    """
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

def plot_series(time, series, ylabel ,format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(True)

def normalize_series(data,min,max):
    """ Function to normalize data columnwise

    Args:
        data (numpy.ndarray): numpy array dataframe
        min (numpy.ndarray): minimum number of each column
        max (numpy.ndarray): maximum number of each column

    Returns:
        numpy.ndarray: normalized series
    """
    return (data - min)/max

def unnormalize_series(data,min,max):
    """ Function to unnormalize data columnwise (in real case scenarios showing a
        price between 0 and 1 is not practical)

    Args:
        data (numpy.ndarray): numpy array dataframe
        min (numpy.ndarray): minimum number of each column
        max (numpy.ndarray): maximum number of each column

    Returns:
        numpy.ndarray: normalized series
    """
    return (data * max) + min

def windowed_dataset(series, batch_size, n_past, n_future, shift=1):
    """ Function to:
        - create dataset from series
        - slice the dataset into appropriate windows
        - flatten the dataset
        - split the dataset into features and labels
        - batch the data

    Args:
        series (_type_): _description_
        batch_size (_type_): _description_
        n_past (_type_): _description_
        n_future (_type_): _description_
        shift (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)

def main(N):
    
    # 2.1

    # Create a pandas dataframe from task 1
    f = open('dict_list.json')
    dict_list = json.load(f)

    # Cast columns into appropriate types
    df = pd.DataFrame(dict_list)
    df = df.astype({
                'product':'string',
                'min_price':'float',
                'max_price':'float',
                'avg_price':'float',
                'origin':'string',
                'distr':'string'
                })
    
    # Statistical summary
    print('\n Descriptive statistical summary of product average prices:\n',df.groupby('product').agg({'describe'})['avg_price'])

    # 2.2

    # Transform the original dataframe to obtain price variables for avocado and create a time variable
    df_avocado = df[df['product']=='Aguacate Hass'].reset_index(drop=True).reset_index()
    df_avocado = df_avocado.rename(columns={"index":"time"})
    df_avocado = df_avocado.drop(columns=['product','origin','distr'])

    # Number of features in the dataset
    N_FEATURES = len(df_avocado.columns)

    # Normalize the data
    data = df_avocado.values
    data_min,data_max = data.min(axis=0),data.max(axis=0)
    data = normalize_series(data, data_min, data_max)

    # Split the data into train and test sets. As test set is pretty important here,
    # we will do a 80/20 split and try to create a model that generalizes well on the test set (Key metrics = mse)
    SPLIT_TIME = int(len(data)*0.8)
    x_train = data[:SPLIT_TIME]
    x_test = data[SPLIT_TIME:]

    # Small batch size hence the small amount of data
    BATCH_SIZE = 16

    # Number of past time steps based on which future observations should be predicted.
    N_PAST = 1

    # Number of future time steps which are to be predicted.
    N_FUTURE = 1

    # Positions from which the window slides to create a new window
    SHIFT = 1

    # Creating a callback to tune the learning rate
    #lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 20))
    # The optimal learning rate was around 0.001

    # Creation of windowed train and test set
    train_set = windowed_dataset(series = x_train, batch_size = BATCH_SIZE,
                                 n_past = N_PAST, n_future = N_FUTURE,
                                 shift = SHIFT)

    test_set = windowed_dataset(series = x_test, batch_size = BATCH_SIZE,
                                n_past = N_PAST, n_future = N_FUTURE,
                                shift = SHIFT)
    
    # Important note: this model is a multi-step forecasting where num inputs = num predictions
    # RNN architecture with a bidirectional layer to take into account past and future information
    # Dropout is used to prevent overfitting
    model = tf.keras.models.Sequential([

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences = True,
                                      input_shape = [N_PAST,N_FEATURES], batch_size = BATCH_SIZE)),
        
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(30, activation='relu'),

        tf.keras.layers.Dense(10, activation='relu'),

        tf.keras.layers.Dense(N_FEATURES)

    ])

    # Number of epochs to train
    EPOCHS = 30

    model.compile(loss = tf.keras.losses.Huber(),
                  optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0015),
                  metrics = ["mse","accuracy"],
                  run_eagerly = True)
    
    history = model.fit(train_set, validation_data = test_set, epochs = EPOCHS) #callbacks = [lr_schedule]) was used to tune the learning rate

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plot_metrics(history, "accuracy")

    # MSE plot
    plt.figure(figsize=(10, 6))
    plot_metrics(history, "mse")

    # Take the last 2 rows we trained with
    starting_data = x_train[SPLIT_TIME-2:] 
    w_start_set = windowed_dataset(series = starting_data, batch_size = BATCH_SIZE,
                                   n_past = N_PAST, n_future = N_FUTURE,
                                   shift = 1)

    # Predict avocado price min and max N days into the future, starting where the train set ends
    for i in range(N):
        if i != N-1:
            temp_pred = history.model.predict(w_start_set)
            temp_pred = temp_pred.reshape(temp_pred.shape[0],-1)
            temp_arr = np.vstack((starting_data[-1],temp_pred))
            w_start_set = windowed_dataset(series = temp_arr[-2:], batch_size = BATCH_SIZE,
                                           n_past = N_PAST, n_future = N_FUTURE,
                                           shift = 1)
            starting_data = temp_arr
        else:
            future_pred = history.model.predict(w_start_set)
            future_pred = future_pred.reshape(future_pred.shape[0],-1)
            future_pred = unnormalize_series(future_pred,data_min,data_max)
            print('Estimated maximum and minimum price of avocado in',N,'days:', round(future_pred[0][2],2),'$,',round(future_pred[0][1],2), '$')
    
    print('\n Metrics of the model in the test set:')
    print(model.evaluate(test_set))

    return model

if __name__ == '__main__':
    N = int(input("Enter number of days to estimate future avocado maximum and minimum price. "))
    model = main(N)
    model.save("avocado_predictor.h5")
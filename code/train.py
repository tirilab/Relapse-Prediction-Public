import os
import numpy as np
import pandas as pd
import scipy
from scipy import signal, stats
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from keras.src.layers.normalization.batch_normalization_v1 import BatchNormalization
import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Flatten, Reshape, Conv1D, MaxPooling1D
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop


def Trainer(X_train, window):

    num_feature = X_train.shape[2]
    filters = 64
    latent = 15   # best # latent features and # filters
    input_dim = (window, num_feature, 1)
    input_img = Input(shape=input_dim)  # adapt this if using `channels_first` image data format
    x = Conv2D(filters, (11, num_feature), activation='leaky_relu', padding='same')(input_img) # select kernel size, 11 generates the best
    x = Flatten()(x)
    x = Dense(latent, activation='leaky_relu')(x)
    x = Dense(units=window*num_feature*filters, activation=tf.nn.relu)(x)
    x = Reshape(target_shape=(window, num_feature, filters))(x)
    decoder = Conv2DTranspose(1, 3, activation='leaky_relu', padding='same')(x)

    ## this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoder)
    autoencoder.summary()

    return autoencoder


def load_data(file_name, norm_cols, window, mode:str):
    
    df = pd.read_csv(file_name, index_col=0).dropna()
    df['sleeping'] = df['sleeping'].astype(int)

    ### only select sleeping or awake data or both
    if mode == 'awake':
        df = df[df['sleeping'] == 0]
    elif mode == 'sleep':
        df = df[df['sleeping'] == 1]

    df = df.drop(columns=['sleeping','sin_t','cos_t'])
    df_train = df[df['split'] == 't']
    df_val = df[df['split'] == 'v']
    df_val_normal = df_val[df_val['label'] == 'normal']
    df_val_relapse = df_val[df_val['label'] == 'relapse']

    ### convert a 4-hour interval into an image
    num_feature = len(norm_cols)
    def pad(df, window, day_index):
        mat = df[df['day_index'] == day_index][norm_cols]
        med = [mat.median().to_numpy()]*(window - len(mat) % window)
        mat = mat.to_numpy()
        mat = np.concatenate([mat, med])
        return mat #.reshape(window, mat.shape[1], -1)
    # np.pad(mat, pad_width=((0,37),(0,0)), mode='constant')

    ### padding
    days = sorted(df_train['day_index'].unique())
    padded = pad(df_train, window, days[0])
    for day in days[1:]:
        padded = np.concatenate([padded,pad(df_train, window, day)], axis=0)
        # print(pad(df_train, window, day))

    days = sorted(df_val_normal['day_index'].unique())
    padded_vn = pad(df_val_normal, window, days[0])
    for day in days[1:]:
        padded_vn = np.concatenate([padded_vn, pad(df_val_normal, window, day)], axis=0)
        # print(pad(df_train, window, day))

    days = sorted(df_val_relapse['day_index'].unique())
    padded_vr = pad(df_val_relapse, window, days[0])
    for day in days[1:]:
        padded_vr = np.concatenate([padded_vr, pad(df_val_relapse, window, day)], axis=0)
        # print(pad(df_train, window, day))

    # print(padded.shape, padded_vn.shape, padded_vr.shape)
    # print(padded.mean(axis=0).round(2), '\n',padded_vn.mean(axis=0).round(2), '\n',padded_vr.mean(axis=0).round(2))
    # print(padded.std(axis=0).round(2), '\n',padded_vn.std(axis=0).round(2), '\n',padded_vr.std(axis=0).round(2))

    ### standardization
    scaler = StandardScaler()
    scaler.fit(padded)
    x_tr = scaler.transform(padded)
    x_vn = scaler.transform(padded_vn)
    x_vr = scaler.transform(padded_vr)
    X_train = x_tr.reshape(-1,window, num_feature, 1)
    X_val_n = x_vn.reshape(-1, window, num_feature, 1)
    X_val_r = x_vr.reshape(-1, window, num_feature, 1)

    print(X_train.shape, X_val_n.shape, X_val_r.shape)
    print(x_tr.mean(axis=0).round(2), '\n', x_vn.mean(axis=0).round(2), '\n', x_vr.mean(axis=0).round(2))
    print(x_tr.std(axis=0).round(2), '\n', x_vn.std(axis=0).round(2), '\n', x_vr.std(axis=0).round(2))

    return X_train, X_val_n, X_val_r

import os
import numpy as np
import pandas as pd
import scipy
from scipy import signal, stats
import datetime
from matplotlib import pyplot as plt
import seaborn as sns
import neurokit2 as nk
import hrvanalysis as hrv
import pyhrv
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE,SpectralEmbedding 
from itertools import chain
from keras.models import load_model
from keras import backend as K

def load_test_data(test_file, data_file, window, train_mode, eval_mode):
    
    df = pd.read_csv(test_file, index_col=0).dropna()

    ## load train data for normalization
    df_t['sleeping'] = df_t['sleeping'].astype(int)
    df_t = pd.read_csv(data_file, index_col=0).dropna()

    norm_cols = ['lin_acc_norm', 'ang_acc_norm', 'heartRate_mean', 'heartRate_max',
          'heartRate_min', 'rRInterval_mean', 'rRInterval_rmssd',
          'rRInterval_sdnn', 'rRInterval_sd1', 'rRInterval_sd2',
          'rRInterval_lombscargle_power_high','rRInterval_lombscargle_power_low']

    ### select only sleeping or awake data or both for train data
    if train_mode == 'awake':
        train = df_t[(df_t['sleeping'] == 0) & (df_t['split'] == 't')]
    elif train_mode == 'sleep':
        train = df_t[(df_t['sleeping'] == 1) & (df_t['split'] == 't')]

    ### select only sleeping or awake data or both for test data
    if eval_mode == 'awake':
        df = df[df['sleeping'] == 0]
    elif eval_mode == 'sleep':
        df = df[df['sleeping'] == 1]

    train = train.drop(columns=['sleeping','sin_t','cos_t','user','split'])
    df = df.drop(columns=['sleeping','sin_t','cos_t'])

    df[df['label'] == 'relapse']['day_index'].unique()
    # print(len(df[df['label'] == 'normal']['day_index'].unique()))
    # print(len(df[df['label'] == 'relapse']['day_index'].unique()))
    df_val_normal = df[df['label'] == 'normal']
    df_val_relapse = df[df['label'] == 'relapse']
    # print(len(df_val_normal['day_index'].unique()))
    # print(len(df_val_relapse['day_index'].unique()))

    ## normalize data
    num_feature = len(norm_cols)
    def pad(df, window, day_index):
        mat = df[df['day_index'] == day_index][norm_cols]
        med = [mat.median().to_numpy()]*(window - len(mat) % window)
        mat = mat.to_numpy()
        mat = np.concatenate([mat, med])
        return mat #.reshape(window, mat.shape[1], -1)
    # np.pad(mat, pad_width=((0,37),(0,0)), mode='constant')

    days = sorted(train['day_index'].unique())
    padded = pad(train, window, days[0])
    # day_index = [days[0]]
    for day in days[1:]:
        padded = np.concatenate([padded,pad(train, window, day)], axis=0)
        # day_index.append(day)
        # print(pad(df_train, window, day))

    days = sorted(df_val_normal['day_index'].unique())
    padded_vn = pad(df_val_normal, window, days[0])
    day_index_vn = [[days[0]]*int(padded_vn.shape[0]/48)]
    for day in days[1:]:
        pad_new = pad(df_val_normal, window, day)
        padded_vn = np.concatenate([padded_vn, pad_new], axis=0)
        day_index_vn.append([day]*int(pad_new.shape[0]/48))
        # print(pad(df_train, window, day))

    days = sorted(df_val_relapse['day_index'].unique())
    padded_vr = pad(df_val_relapse, window, days[0])
    day_index_vr = [[days[0]]*int(padded_vr.shape[0]/48)]
    for day in days[1:]:
        pad_new = pad(df_val_relapse, window, day)
        padded_vr = np.concatenate([padded_vr, pad_new], axis=0)
        day_index_vr.append([day]*int(pad_new.shape[0]/48))
        # day_index_vr.append(day)
        # print(pad(df_train, window, day))

    # print(padded_vn.shape, padded_vr.shape)
    # print(padded_vn.mean(axis=0).round(2), '\n',padded_vr.mean(axis=0).round(2))
    # print(padded_vn.std(axis=0).round(2), '\n',padded_vr.std(axis=0).round(2))

    scaler = StandardScaler()
    scaler.fit(padded)
    x_tr = scaler.transform(padded)
    x_vn = scaler.transform(padded_vn)
    x_vr = scaler.transform(padded_vr)
    X_train = x_tr.reshape(-1, window, num_feature, 1)
    X_val_n = x_vn.reshape(-1, window, num_feature, 1)
    X_val_r = x_vr.reshape(-1, window, num_feature, 1)
    print(X_train.shape, X_val_n.shape, X_val_r.shape)
    # print(x_tr.mean(axis=0).round(2), '\n', x_vn.mean(axis=0).round(2), '\n', x_vr.mean(axis=0).round(2))
    # print(x_tr.std(axis=0).round(2), '\n', x_vn.std(axis=0).round(2), '\n', x_vr.std(axis=0).round(2))
    # X_val_n_day_index = df_val_normal['day_index'].to_numpy()
    # X_val_r_day_index = df_val_relapse['day_index'].to_numpy()

    return X_train, X_val_n, X_val_r, day_index_vn, day_index_vr

def evaluation(y_true, pred_label):
    print(f'accuracy score: {accuracy_score(y_true, pred_label)}')
    cf_mat = confusion_matrix(y_true, pred_label)
    print('Confusion matrix')
    print(cf_mat)
    n = len(y_true)
    ratio = len(y_true[y_true==1])/len(y_true)
    n_0 = int((1-ratio) * n)
    n_1 = int(ratio * n)

    print(f'class 0 accuracy: {cf_mat[0][0]/n_0}')
    print(f'class 1 accuracy: {cf_mat[1][1]/n_1}')

def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

def roc_metric(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    return specificity, sensitivity


def eval_results(input_latent, labels):
    pred_gm = GaussianMixture(n_components=2, random_state=0).fit_predict(input_latent)
    pred_km = KMeans(n_clusters=2, n_init='auto', random_state=0).fit_predict(input_latent)
    y_true = np.array(labels)
    # evaluation(y_true, pred_km)
    # plot_roc_curve(y_true, pred_km)
    p_gm, r_gm, _ = precision_recall_curve(y_true, pred_gm)
    p_km, r_km, _ = precision_recall_curve(y_true, pred_km)
    auprc_gm = auc(r_gm, p_gm)
    auprc_km = auc(r_km, p_km)
    auroc_gm = roc_auc_score(y_true, pred_gm)
    auroc_km = roc_auc_score(y_true, pred_km)
    auprc_base = np.mean(y_true)
    # print(f'GMM auPRC score: {round(auprc_gm,4)}')
    # print(f'KMeans auPRC score: {round(auprc_km,4)}')
    # print(f'GMM auROC score: {round(auroc_gm,4)}')
    # print(f'KMeans auROC score: {round(auroc_km,4)}')
    # print(f'baseline auPRC {round(np.mean(y_true),4)}')
    f1_gm = 2/(1/auprc_gm + 1/auroc_gm)
    f1_km = 2/(1/auprc_km + 1/auroc_km)
    return auprc_gm, auprc_km, auroc_gm, auroc_km, auprc_base, f1_gm, f1_km, pred_gm, pred_km, y_true

def eval_results_daily(input_latent, labels, day_index):
    pred_gm = GaussianMixture(n_components=2, random_state=0).fit_predict(input_latent)
    pred_km = KMeans(n_clusters=2, n_init='auto', random_state=0).fit_predict(input_latent)
    y_true = np.array(labels)
    daily_label = []
    daily_pred_gm = []
    daily_pred_km = []
    for day in np.unique(day_index):
        daily_label.append(int(np.mean(y_true[np.argwhere(day_index == day)])))
        daily_pred_gm.append(np.mean(pred_gm[np.argwhere(day_index == day)]))
        daily_pred_km.append(np.mean(pred_km[np.argwhere(day_index == day)]))
    # evaluation(y_true, pred_km)
    # plot_roc_curve(y_true, pred_km)

    fpr, tpr, thresholds = roc_curve(daily_label, daily_pred_km)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    # print(optimal_threshold)

    p_gm, r_gm, _ = precision_recall_curve(daily_label, np.array(daily_pred_gm)>0)
    p_km, r_km, _ = precision_recall_curve(daily_label, np.array(daily_pred_km)>0)

    # f1_scores = 2 * (p_gm * r_gm) / (p_gm + r_gm)
    # optimal_threshold_pr = thresholds[np.argmax(f1_scores)]

    auprc_gm = auc(r_gm, p_gm)
    auprc_km = auc(r_km, p_km)
    auroc_gm = roc_auc_score(daily_label, np.array(daily_pred_gm))
    auroc_km = roc_auc_score(daily_label, np.array(daily_pred_km))
    auprc_base = np.mean(y_true)

    f1_gm = 2/(1/auprc_gm + 1/auroc_gm)
    f1_km = 2/(1/auprc_km + 1/auroc_km)

    return auprc_gm, auprc_km, auroc_gm, auroc_km, auprc_base, f1_gm, f1_km, daily_pred_gm, daily_pred_km, daily_label

def flatten_chain(matrix):
  return list(chain.from_iterable(matrix))

def reconstruction(X_val_n, X_val_r):

    X = np.concatenate([X_val_n, X_val_r])
    reconstructed_model = load_model(model_path)

    get_last_layer_output = K.function(
      [reconstructed_model.layers[0].input], # param 1 will be treated as layer[0].output
      [reconstructed_model.layers[3].output]) # and this function will return output from 3rd layer

    # here X is param 1 (input) and the function returns output from layers[3]
    out_img = get_last_layer_output([X])[0]

    return out_img

def generate_evaluation_csv(data_dir, model_dir, test_dir, data_file, model_path, test_file, train_mode:str, eval_mode:str):

    auprc_list = []
    auroc_list = []
    auprc_base_list = []
    f1_list = []
    result_best_index = []
    pred_list = []
    true_list = []
    
    window = 48
    X_train, X_val_n, X_val_r, X_val_n_day_index, X_val_r_day_index = load_test_data(test_file, data_file, window)

    out_img = reconstruction(X_val_n, X_val_r)
    labels = [0]*X_val_n.shape[0] + [1]*X_val_r.shape[0]
    val_day_index = flatten_chain(X_val_n_day_index) + flatten_chain(X_val_r_day_index)

    X_pca = PCA(n_components=4, random_state=0).fit_transform(out_img)
    X_tsne = TSNE(n_components=3, learning_rate='auto',
                  init='pca', perplexity=10, random_state=0).fit_transform(out_img)

    se = SpectralEmbedding(n_components=3, n_neighbors=10, random_state=0)
    f = se.fit_transform(out_img)

    auprc_gm0, auprc_km0, auroc_gm0, auroc_km0, auprc_base, f1_gm0, f1_km0, pred_gm0, pred_km0, y_true = eval_results_daily(out_img, labels, val_day_index)
    auprc_gm1, auprc_km1, auroc_gm1, auroc_km1, auprc_base, f1_gm1, f1_km1, pred_gm1, pred_km1, y_true = eval_results_daily(X_pca[:,:2], labels, val_day_index)
    auprc_gm2, auprc_km2, auroc_gm2, auroc_km2, auprc_base, f1_gm2, f1_km2, pred_gm2, pred_km2, y_true = eval_results_daily(X_tsne[:,:2], labels, val_day_index)
    auprc_gm3, auprc_km3, auroc_gm3, auroc_km3, auprc_base, f1_gm3, f1_km3, pred_gm3, pred_km3, y_true = eval_results_daily(f[:,:2], labels, val_day_index)

    auprcs = [auprc_gm0, auprc_gm1, auprc_gm2, auprc_gm3, auprc_km0, auprc_km1, auprc_km2, auprc_km3]
    aurocs = [auroc_gm0, auroc_gm1, auroc_gm2, auroc_gm3, auroc_km0, auroc_km1, auroc_km2, auroc_km3]
    preds = [pred_gm0, pred_gm1, pred_gm2, pred_gm3, pred_km0, pred_km1, pred_km2, pred_km3]
    f1s = [f1_gm0, f1_gm1, f1_gm2, f1_gm3, f1_km0, f1_km1, f1_km2, f1_km3]

    user_result_df = pd.DataFrame({'auprc':auprcs, 'auroc':aurocs, 'f1': f1s})
    result_best = user_result_df['f1'].idxmax()

    auprc_best = user_result_df.loc[result_best, 'auprc']
    auroc_best = user_result_df.loc[result_best, 'auroc']
    f1_best = user_result_df.loc[result_best, 'f1']

    auprc_list.append(auprc_best)
    auroc_list.append(auroc_best)
    auprc_base_list.append(auprc_base)
    result_best_index.append(result_best)
    f1_list.append(f1_best)
    pred_list.append(preds[result_best])
    true_list.append(y_true)

    result_df = pd.DataFrame(data={'auprc': auprc_list, 'auroc': auroc_list, 'auprc_base': auprc_base_list,
                               'f1': f1_list, 'result_best': result_best_index, 'pred': pred_list,
                               'y_true':true_list})

    return result_df

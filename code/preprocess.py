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

### load files
def load_files(data_dir, file_info):
    
    split = file_info['split']
    label = file_info['label']
    user = file_info['user']
    file_name = file_info['file_name']
    
    if split == 'train/' and label == 'relapse':
        raise Exception("A training set doesn't contain relapse events.")
    else:
        data_path = data_dir + user + split + label
        days = sorted([day for day in sorted(os.listdir(data_path)) if day[0].isnumeric()])
        file_paths = [data_path + day + file_name for day in days]
    
    return file_paths

# functions to filter out "bad" RR intervals in order to generate NN intervals
#### Filter 1 ####
def RR_filter_1(RR):
    RR_nan = np.copy(RR)       # make copy of RR intervals
    # RR_filtered = np.copy(RR)  # make copy of RR intervals
        
    #  filter based upon abs RR
    index=np.argwhere(RR>np.median(RR)*2.5)
    if len(index)>0:
        # RR_filtered=np.delete(RR_filtered,index) # remove bad RR intervals
        RR_nan[index]=np.nan   # keep track of location of RR intervals by making it NAN
 
    index = np.argwhere(np.isnan(RR_nan)) # index of nan values

    return RR_nan, index

#### Filter 2 ####
def RR_filter_2(RR):
    RR_nan = np.copy(RR)       # make copy of RR intervals
    RR_filtered = np.copy(RR)  # make copy of RR intervals
       
    #  filter based upon percent difference (from median)
    for k in range(0,2):
        p = percent_difference(RR_filtered)
        if np.max(np.abs(p))>40:
            index = np.argwhere(np.abs(p)>40)
            # RR_filtered = np.delete(RR_filtered,index)
            RR_nan[index] = np.nan
            
    index = np.argwhere(np.isnan(RR_nan)) # index of nan values
    return RR_nan, index

def percent_difference(RR):
    mvalue=signal.medfilt(RR,5)
    medianValue=np.median(RR)
    p=(RR-mvalue)/medianValue*100
    return p

#### Filter 3 ####
def RR_filter_3(RR):
    RR_nan = np.copy(RR)       # make copy of RR intervals
#     RR_filtered = np.copy(RR)  # make copy of RR intervals
       
    #  spike detection (higher)
    diffr=np.diff(RR_filtered)
    index=np.argwhere(diffr>np.median(RR_filtered)*0.4)
    if len(index)>0:
        index=np.argwhere(RR>=min(RR[index+1]))
#         RR_filtered=np.delete(RR_filtered,index) # remove bad RR intervals
        RR_nan[index]=np.nan   # keep track of location of RR intervals by making it NAN   
    
    index = np.argwhere(np.isnan(RR_nan)) # index of nan values
            
    return RR_nan, index

#### Filter Ectopic ####
def filter_ectopic(RR, ectopic_method):
    # Remove ectopic beats from signal (ectopic beats become NANs)
    # ectopic_method : method to use to remove ectopic beats (malik, kamath, karlsson, acar); default='acar'

    NN_int_ms_ectopicNAN = hrv.remove_ectopic_beats(rr_intervals=RR, method=ectopic_method)
    
    return NN_int_ms_ectopicNAN

def nni_pipeline(RR_int_ms, ectopic_method="acar",*args,**kwargs):
    # inputs:
        # ecg : ecg dataframe
        # lead : which lead to process (can be 1 to 7); default=1
        # start : first sample of ecg for processing; default=0
        # end : last sample of ecg for processing; default=last sample of ecg waveform
        # fs : sampling rate in Hz; default=240
        # ectopic_method : method to use to remove ectopic beats (malik, kamath, karlsson, acar); default='acar'

    # calculate RR intervals
    # RR_int_ms = find_RR_ms_mimic(ecg, fs, lead, start, end)

    # filter RR intervals
    RR_nan_1, index_1 = RR_filter_1(RR_int_ms) # first filter
    RR_nan_2, index_2 = RR_filter_2(RR_nan_1) # second filter
    RR_nan_3, index_3 = RR_filter_2(RR_nan_2) # third filter

    # Remove ectopic beats from signal (ectopic beats become NANs)
    NN_int_ms_ectopicNAN = filter_ectopic(RR_int_ms, ectopic_method)
    
    return NN_int_ms_ectopicNAN

##### Imputation
def find_miss_valid(df, miss_hour, minute_interval):
    
    # miss_tot = calculate_missing_hour_3(df)
    timecol = pd.to_datetime(df['timecol'],format= '%H:%M:%S')
    # timecol.resample('5T').mean()
    time_diff = timecol.diff()
    gaps = time_diff[time_diff >= pd.Timedelta(minutes=minute_interval)]
    gaps = gaps[time_diff <= pd.Timedelta(minutes=miss_hour*60)]
    
    return gaps

def median_interpolation_previous(df):

    df['timecol'] = pd.to_datetime(df['timecol'],format= '%H:%M:%S').dt.time
    start = pd.to_datetime(str(df['timecol'].min()))
    end = pd.to_datetime('23:59:55')
    dates = pd.date_range(start=start, end=end, freq='5S').time

    df = df.set_index('timecol').reindex(dates).reset_index().reindex(columns=df.columns)
    cols = df.columns.difference(['val'])
    # df[cols] = df[cols].ffill()  # nearest neighbors: forward fill

    na_idx = df.iloc[df['sleeping'].isnull().to_numpy(), :].index
    start_idx = na_idx[0]
    end_idx = df.iloc[df['sleeping'].isnull().to_numpy(), :].index[-1]

    impute_interval_hour = 3
    impute_n_row = int(impute_interval_hour * 3600/5)
    if start_idx >= impute_n_row: 
        previous_median = df.iloc[start_idx-impute_n_row:start_idx-1, ~df.columns.isin(["timecol"])].median()
        noise_shape = df.iloc[start_idx:end_idx+1, df.columns.isin(["heartRate","rRInterval"])].shape
        noise = np.random.rand(noise_shape[0], noise_shape[1])
        df.iloc[start_idx:end_idx+1, ~df.columns.isin(["timecol"])] = previous_median 
        df.iloc[start_idx:end_idx+1, df.columns.isin(["heartRate","rRInterval"])] += noise
    else:
        noise_shape = df.iloc[start_idx:end_idx+1, df.columns.isin(["heartRate","rRInterval"])].shape
        noise = np.random.rand(noise_shape[0], noise_shape[1])
        med = df.iloc[0:start_idx-1,~df.columns.isin(["timecol"])].median()
        df.iloc[na_idx, ~df.columns.isin(["timecol"])] = med
        df.iloc[start_idx:end_idx+1, df.columns.isin(["heartRate","rRInterval"])] += noise
    return df

valid_ranges = {
    "acc_X" : (-19.6, 19.6),
    "acc_Y" : (-19.6, 19.6),
    "acc_Z" : (-19.6, 19.6),
    "gyr_X" : (-573, 573),
    "gyr_Y" : (-573, 573),
    "gyr_Z" : (-573, 573),
    "heartRate" : (0.1, 255),
    "rRInterval" : (0.1, 2000),
}

def set_valid_range(df1, valid_ranges=valid_ranges):   
    df1.loc[(df1['acc_X'] < valid_ranges['acc_X'][0]) | (df1['acc_X'] >= valid_ranges['acc_X'][1]), 'acc_X'] = np.nan
    df1.loc[(df1['acc_Y'] < valid_ranges['acc_Y'][0]) | (df1['acc_Y'] >= valid_ranges['acc_Y'][1]), 'acc_Y'] = np.nan
    df1.loc[(df1['acc_Z'] < valid_ranges['acc_Z'][0]) | (df1['acc_Z'] >= valid_ranges['acc_Z'][1]), 'acc_Z'] = np.nan
    df1.loc[(df1['gyr_X'] < valid_ranges['gyr_X'][0]) | (df1['gyr_X'] >= valid_ranges['gyr_X'][1]), 'gyr_X'] = np.nan
    df1.loc[(df1['gyr_Y'] < valid_ranges['gyr_Y'][0]) | (df1['gyr_Y'] >= valid_ranges['gyr_Y'][1]), 'gyr_Y'] = np.nan
    df1.loc[(df1['gyr_Z'] < valid_ranges['gyr_Z'][0]) | (df1['gyr_Z'] >= valid_ranges['gyr_Z'][1]), 'gyr_Z'] = np.nan
    df1.loc[(df1['heartRate'] <= valid_ranges['heartRate'][0]) | (df1['heartRate'] >= valid_ranges['heartRate'][1]), 'heartRate'] = np.nan
    df1.loc[(df1['rRInterval'] <= valid_ranges['rRInterval'][0]) | (df1['rRInterval'] >= valid_ranges['rRInterval'][1]), 'rRInterval'] = np.nan
    return df1

####### metrics #######
def compute_mean(x:pd.Series, step=60):
    # don't need to skip na for nanmean: skipna (default) = True
    output = x.groupby(x.index // step).mean()
    return output

def lin_norm(df):
    norm = (df['acc_X']**2 + df['acc_Y']**2 + df['acc_Z']**2).pow(1/2)
    return norm.mean()

def ang_norm(df):
    norm = (df['gyr_X']**2 + df['gyr_Y']**2 + df['gyr_Z']**2).pow(1/2)
    return norm.mean()

def poincare_sd1(nni):   
    # Prepare Poincaré data
    x1 = np.asarray(nni[:-1])
    x2 = np.asarray(nni[1:])
    # SD1 & SD2 Computation
    sd1 = np.std(np.subtract(x1, x2) / np.sqrt(2))
     
    # rri_valid = df1['rRInterval'].dropna()
    # sdnn = pyhrv.time_domain.sdnn(rri_valid)[0]
    # sdsd = pyhrv.time_domain.sdsd(rri_valid)[0]
    # sd1 = (0.5*sdsd**2)**(1/2)
    # sd2 = (2*sdnn**2 - 0.5*sdsd**2)**(1/2)
    # print(sd1, sd2)
    return sd1

def poincare_sd2(nni):
    x1 = np.asarray(nni[:-1])
    x2 = np.asarray(nni[1:])
    sd2 = np.std(np.add(x1, x2) / np.sqrt(2))
    return sd2

def lombscargle_power_high(nni):
    # high frequencies
    l = 0.15 * np.pi /2
    h = 0.4 * np.pi /2
    freqs = np.linspace(l, h, 10)
    nni = nni.dropna()
    hf_lsp = scipy.signal.lombscargle(nni.to_numpy(), nni.index.to_numpy(), freqs, normalize=True)
    return np.trapz(hf_lsp, freqs)

def lombscargle_power_low(nni):
    # low frequencies
    l = 0.04 * np.pi /2
    h = 0.15 * np.pi /2
    freqs = np.linspace(l, h, 10)
    # nni = nni[nni > 0]
    lf_lsp = scipy.signal.lombscargle(nni.to_numpy(), nni.index.to_numpy(), freqs, normalize=True)
    return np.trapz(lf_lsp, freqs)

def time_encoding(df):
    # Compute the sin and cos of timestamp (we have 12*24=288 5-minutes per day if no data is missing)
    h = [df['timecol'][i].hour for i in range(len(df))]
    m = [df['timecol'][i].minute for i in range(len(df))]
    time_value = np.add(np.multiply(h, 60), m)
    print(time_value)
    df['sin_t'] = np.sin(time_value*(2.*np.pi/(60*24)))
    df['cos_t'] = np.cos(time_value*(2.*np.pi/(60*24)))
    return sin_t, cos_t

def rmssd(x):
    x = x.dropna()
    try:
        rmssd = pyhrv.time_domain.rmssd(x)[0]
    except (ZeroDivisionError, ValueError):
        rmssd = np.nan
    return rmssd

def sdnn(x):
    x = x.dropna()
    try:
        sdnn = pyhrv.time_domain.sdnn(x)[0]
    except (ZeroDivisionError, ValueError):
        sdnn = np.nan
    return sdnn


#### prepare feature files as input data to the autoencoder training ####
def extract_user_features(data_path, patient, split, label, phase, file_name, frequency:str):
    print(f'{data_path}{patient}{split}{label}{phase}{file_name}')
    df = pd.read_csv(f'{data_path}{patient}{split}{label}{phase}{file_name}')

    # convert RR interval to normal-to-normal intervals (RRI -> NNI)
    df['rRInterval'] = nni_pipeline(df['rRInterval'])
    df = set_valid_range(df)
    df = df.dropna()

    if len(df) > 0:
        # find missing time intervals (less than and more than 3 hours)
        timecol = pd.to_datetime(df['timecol'],format= '%H:%M:%S')
        time_diff = timecol.diff()
        large_gaps = time_diff[time_diff > pd.Timedelta(seconds=3*3600)]
        small_gaps = time_diff[(pd.Timedelta(seconds=5*60) < time_diff)]
        small_gaps = small_gaps[time_diff <= pd.Timedelta(seconds=3*3600)]

        if len(large_gaps) > 0:
            gap_idx = np.argwhere(df.index == large_gaps.index[0])[0][0]
            gap_loc_start = df.iloc[gap_idx - 1]['timecol']
            gap_loc_end = df.iloc[gap_idx]['timecol']

            df['timecol'] = pd.to_datetime(df['timecol'],format= '%H:%M:%S').dt.time
            start1 = pd.to_datetime(str(df['timecol'].min()))
            end1 = pd.to_datetime(str(gap_loc_start))
            dates1 = pd.date_range(start=start1, end=end1, freq='5S').time

            start2 = pd.to_datetime(str(gap_loc_end))
            end2 = pd.to_datetime(str(df['timecol'].max()))
            dates2 = pd.date_range(start=start2, end=end2, freq='5S').time

            dates = np.concatenate([dates1, dates2], axis=0)

        else:
            df['timecol'] = pd.to_datetime(df['timecol'],format= '%H:%M:%S').dt.time
            start = pd.to_datetime(str(df['timecol'].min()))
            end = pd.to_datetime(str(df['timecol'].max()))
            dates = pd.date_range(start=start, end=end, freq='5S').time

        df = df.set_index('timecol').reindex(dates).reset_index().reindex(columns=df.columns)
        cols = df.columns.difference(['val'])
        df.loc[:,cols[:-1].to_list()] = df.loc[:,cols[:-1].to_list()].fillna(df.loc[:,cols[:-1].to_list()].median())
        df['rRInterval'] += np.random.rand(len(df))

        # grouper function to take average of every 5-min interval
        df['DateTime'] = df['timecol'].apply(lambda t: datetime.datetime.combine(datetime.datetime.today(), t))
        df['day_index'] = [phase]*len(df)
        df_linacc = df.groupby([df['day_index'],pd.Grouper(key='DateTime',freq=frequency)]).apply(lin_norm)
        df_angacc = df.groupby([df['day_index'],pd.Grouper(key='DateTime',freq=frequency)]).apply(ang_norm)
        df_hrm = df.groupby([df['day_index'],
                              pd.Grouper(key='DateTime',freq=frequency)]).agg({'heartRate': np.nanmean,
                                                                            'rRInterval':  [np.nanmean, rmssd, sdnn, poincare_sd1, poincare_sd2,
                                                                                            lombscargle_power_high, lombscargle_power_low]})
        df = pd.concat([df_linacc, df_angacc, df_hrm], axis=1)
        df = df.reset_index()

        h = df['DateTime'].dt.hour
        m = df['DateTime'].dt.minute
        time_value = h*60 + m
        df['sin_t'] = np.sin(time_value*(2.*np.pi/(60*24)))
        df['cos_t'] = np.cos(time_value*(2.*np.pi/(60*24)))

        # drop datetime column
        df = df.drop(columns=['DateTime'])

        # rename columns
        new_column_names = ['day_index', 'lin_acc_norm', 'ang_acc_norm',
                            'heartRate_mean', 'rRInterval_mean', 'rRInterval_rmssd', 'rRInterval_sdnn',
                            'rRInterval_sd1', 'rRInterval_sd2',
                            'rRInterval_lombscargle_power_high', 'rRInterval_lombscargle_power_low',
                            'sin_t', 'cos_t']

        df.columns = new_column_names

        # save df
        os.makedirs(f'{features_path}/{patient}/{split}/{label}/{phase}', exist_ok=True)
        df.to_csv(f'{features_path}/{patient}/{split}/{label}/{phase}/features.csv')
        print('Saved features for patient {} and phase {}'.format(patient, phase))

    else:
        print('abnormal signals recorded for the day')

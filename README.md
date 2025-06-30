#  Relapse Prediction using Wearable Data through Convolutional Autoencoders and Clustering for Patients with Psychotic Disorders 

April Yujie Yan, BS, MBI, Traci Jenelle Speed, MD, PhD, Casey Overby Taylor, PhD

Relapse of psychotic disorders occurs commonly even after appropriate treatment. Digital phenotyping becomes essential to achieve remote monitoring for mental conditions. We applied a personalized approach using neural-network-based anomaly detection and clustering to predict relapse for patients with psychotic disorders. We used a dataset provided by e-Prevention grand challenge (SPGC), containing physiological signals for 10 patients monitored over 2.5 years (relapse events: 560 vs. non-relapse events: 2139) [1,2]. We created 2-dimensional multivariate time-series profiles containing activity and heart rate variability metrics, extracted latent features via convolutional autoencoders, and identified relapse clusters. Our model showed promising results compared to the 1st place of SPGC (area under precision-recall curve = 0.711 vs. 0.651, area under receiver operating curve = 0.633 vs. 0.647, harmonic mean = 0.672 vs. 0.649) and added to existing evidence of data collected during sleep being more informative in detecting relapse. Our study demonstrates the potential of unsupervised learning in identifying abnormal behavioral changes in patients with psychotic disorders using objective measures derived from granular, long-term biosignals collected by unobstructive wearables. It contributes to the first step towards determining relapse-related biomarkers that could improve predictions and enable timely interventions to enhance patients’ quality of life.   

**Yan, A.Y., Speed, T.J. & Taylor, C.O. Relapse prediction using wearable data through convolutional autoencoders and clustering for patients with psychotic disorders. Sci Rep 15, 18806 (2025). https://doi.org/10.1038/s41598-025-03856-1**

### Dataset 
The e-Prevention SPGC 2023 [1,2] provided a public dataset containing physiological signals of 10 patients with psychotic disorders, recorded in Samsung Gear S3 smartwatches for a monitoring period of 6 months. Physiological biosignals included users’ linear and angular accelerations (20Hz), heart rate and RR intervals (5Hz), sleeping status, and steps. Daily relapse status was annotated by clinicians by reviewing their monthly assessments and communication with their physicians and/or family members. Data were split into train (only non-relapsed state), validation (both states with each labeled relapse and non-relapse), and test (both states but unlabeled) sets. Data have been made publicly available and can be downloaded here: https://robotics.ntua.gr/eprevention-sp-challenge/. 


#### Measurements in raw data files:

`data.csv`: there is a data file for daily recording

- `acc_X`, `acc_Y`, `acc_Z`:linear acceleration from the accelerometer (Valid range: [-19.6, 19,6])
- `gyr_X`, `gyr_Y`, `gyr_Z`: angular velocity from the gyroscope (Valid range: [-573, 573])
- `heartRate`, `rRInterval`: beats-per-minute and R-R interval from PPG. Valid values are > 0.

- `timecol`: timestamp inside the day

- `sleeping`: 0 if the user is awake, 1 if they are sleeping


### What are in this repository
```
RelapsePredictionPublic/ 
    │- code/
        |- analysis.ipynb # demo of EDA and statisitical analysis
        |- main.ipynb # demo of running the whole pipeline
        |- preprocess.py # helper function for preprocessing raw data 
        |- stats.py # helper function for statistical analysis 
        |- train.py # helper function for preparing input data 
        |- utils.py # helper function for clustering and evaluation 
    │- feature/ # this folder contains processed data. We provide a sample organization from User_00's data. Files should be arranged in the following way by running our code for preprocessing:
        |- User_00/
          |- train/
            |- non-relpase/
              |- Day_{N}/
                |- feature.csv
          |- val/
            |- non-relpase/
              |- Day_{N}/
                |- feature.csv
            |- relapse/
              |- Day_{N}/
                |- feature.csv
          |- test/
              |- Day_{N}/
                |- feature.csv
    │- input/ # processed input data files for each patient (combined from data in feature/), for training. We provide a sample file from User_00
        |- input_v2_user00.csv
    |- test # processed input data files for each patient (combined from data in feature/), for testing. We provide a sample file from User_00
        |- test_v2_user00.csv
    │- save_model/ # a folder that contains saved models. e.g., `latent_k{kernel_size}_{training_mode}_{patient ID}.keras`. `training_mode` indicates how the autoencoder was trained, based on sleep data only, awake data only, or both.
        |- `latent_k11_sleep_user00.keras`
        |- `latent_k11_awake_user00.keras`
        |- `latent_k11_both_user00.keras`
    |- results/ # a folder to save validation results. e.g., result_{filter_number}{latent_dimension}_concat_k{kernel_size}_{train_mode}_{eval_mode}.csv
    │- requirements.txt # Dependencies
    │- README.md # Documentation
    │- .gitignore # Ignore unnecessary files
    │- LICENSE
```

### Run Demonstration 


Clone the repository. Make sure you have the installed packages in `requirements.txt`:
```
git clone https://github.com/{yourusername}/Relapse-Prediction-Public.git
```
```
pip install -r requirements.txt
```

Download zipped data from the link above and when you unzip it, they should be structured in the following way:

```
training_data/
    |- User00/
      |- train/
        |- non-relapse/
          |- Day_{N}/
            |- data.csv
            |- step.csv
      |- val/
        |- non-relapse/
          |- Day_{N}/
            |- data.csv
            |- step.csv
        |- relapse/
          |- Day_{N}/
            |- data.csv
            |- step.csv    
test_data/
    |- User00/
      |- test/
        |- Day_{N}/
          - data.csv
          - step.csv 
```

`analysis.ipynb`: This files contains details of exploratory data analysis and shows example outputs/plots from Kolmogorov–Smirnov test. 

`main.ipynb`: In this file, we included code snippets for the whole pipeline including data extraction (i.e., extract features for each non-overlapping five-minute interval), autoencoder training (i.e., prepare data as 2D profiles and train autoencoder to reconstruct them), clustering (i.e., cluster relapse status based on latent features), and result generation (i.e., generate evaluation result .csv for a pre-specified training setting). 

1. Data Preprocessing

In this section, you can process raw data by summarizing desired metrics into non-overlapping 5-min intervals. You need to specify the data folder (e.g., train/val/test). 

- Specify path to the data folder. E.g., code below shows we selected the data folder corresponding to `User_00/test/`
```
data_path = '../train_data/' 
splits = ['train/', 'val/', 'test/']
labels = ['relapse/','non-relapse/', '']

patient = patients[0]
split = splits[-1]
label = labels[-1]
```

The output of this step will be save in `feature/` as a series of individual files of daily recordings with metrics summarized into non-overlapping 5-min intervals. P.S. if there isn't one empty folder, please create one yourself. **Or you can go ahead and use the ready-to-input data shared in `input/` processed and combined by us.**

2. Autoencoder Training

In this section, for each patient/user, you can train autoencoders to extract latent features from the middle layer by loading and normalizing input data and optimizing autoencoder models. You can also visualize the reconstructed 2D profile by visualizing the final output. For studying the effect of sleeping status on autoencoder training and clustering, we performed stratified analysis based on sleep-only, awake-only, and all data. 

- You need to specify the training mode, whether you want to use sleep, awake, or both data to train an autoencoder.  

```
train_mode = 'sleep' # specify data used for CAE tranining: sleep, awake, or both 
```

The output of this step will be saved in `save_model/`, e.g., `latent_k11_sleep_00.keras`

3. Clustering

In this section, you can use perform clustering regarding the relapse status. The number of clusters is always 2 because we assume there are only two states, relapse vs. non-relapse. Through clustering, you can compute the prediction on a per day basis for further evaluation.   

- You need to specify the autoencoder training and clustering mode.

```
train_mode, eval_mode = 'sleep', 'both' # e.g., "sleep","sleep"; "awake","both"
model_path = model_dir + f'latent_k11_{train_mode}_0{i}.keras' # i represents patient id
save_dir = f'../results/result_6415_concat_k11_{train_mode}_{eval_mode}.csv'
```

The output of this step will be saved in `results/` as a csv file containing evaluation results and predicted daily label for each model, e.g., `result_6415_concat_k11_sleep_both.csv` is for CAE training using sleep-only data and clustering on all data. Note that our default code will run through for all patients but we only provide User_00's data as sample in this repository. If you want to re-generate result files, you will need to specify the patient id. 

### Reference
1. Athanasia Z, Panagiotis PF, Christos G, Efthymiou N, Maragos P, Menychtas A, et al. E-Prevention: Advanced support system for monitoring and relapse prevention in patients with psychotic disorders analyzing long-term multimodal data from wearables and video captures. 2022;22(19):7544–4.

2. Athanasia Z et al. Person Identification and Relapse Detection from Continuous Recordings of Biosignals Challenge: Overview and Results. IEEE Open Journal of Signal Processing. 2024; 5:641-651. 10.1109/OJSP.2024.3376300.

### Contact
If you have any questions, please get in touch with April Yan, yyan67@jhu.edu.

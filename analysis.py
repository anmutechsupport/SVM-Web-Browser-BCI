import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes, AggOperations, WindowFunctions, NoiseTypes
from scipy.signal import butter, lfilter, lfilter_zi
from tqdm import tqdm 

def vectorize(df, fs, filtering=False, streaming=False):

    if streaming == True:
        index = len(df)
        feature_vectors = []
        if filtering == True:

            DataFilter.perform_bandpass(df[:], fs, 15.0, 6.0, 4,
                                FilterTypes.BESSEL.value, 0)
            DataFilter.remove_environmental_noise(df[:], fs, NoiseTypes.SIXTY.value)

        for y in range(0,index,fs):

            f, Pxx_den = signal.welch(df[y:y+fs], fs=fs, nfft=256) #simulated 4 point overlap

            ind_delta, = np.where(f < 4)
            meanDelta = np.mean(Pxx_den[ind_delta], axis=0)
            # Theta 4-8
            ind_theta, = np.where((f >= 4) & (f <= 8))
            meanTheta = np.mean(Pxx_den[ind_theta], axis=0)
            # Alpha 8-12
            ind_alpha, = np.where((f >= 8) & (f <= 12))
            meanAlpha = np.mean(Pxx_den[ind_alpha], axis=0)
            # Beta 12-30
            ind_beta, = np.where((f >= 12) & (f < 30))
            meanBeta = np.mean(Pxx_den[ind_beta], axis=0)
            # Gamma 30-100+
            ind_Gamma, = np.where((f >= 30) & (f < 40))
            meanGamma = np.mean(Pxx_den[ind_Gamma], axis=0)

            feature_vectors.insert(y, [meanDelta, meanTheta, meanAlpha, meanBeta, meanGamma])

        powers = np.log10(np.asarray(feature_vectors))

        powers = powers.reshape(5)
        return powers
    
    else:
        index, ch = df.shape[0], df.shape[1]
        feature_vectors = [[], []]

        for x in tqdm(range(ch)):

            if filtering == True:

                DataFilter.perform_bandpass(df[:, x], fs, 15.0, 6.0, 4,
                                    FilterTypes.BESSEL.value, 0)
                DataFilter.remove_environmental_noise(df[:, x], fs, NoiseTypes.SIXTY.value)

            for y in range(fs,index,fs):

                f, Pxx_den = signal.welch(df[y-fs:y, x], fs=fs, nfft=256) #simulated 4 point overlap
                # plt.semilogy(f, Pxx_den)
                # plt.ylim([0.5e-3, 1])
                # plt.xlabel('frequency [Hz]')
                # plt.ylabel('PSD [V**2/Hz]')
                # plt.show()

                ind_delta, = np.where(f < 4)
                meanDelta = np.mean(Pxx_den[ind_delta], axis=0)
                # Theta 4-8
                ind_theta, = np.where((f >= 4) & (f <= 8))
                meanTheta = np.mean(Pxx_den[ind_theta], axis=0)
                # Alpha 8-12
                ind_alpha, = np.where((f >= 8) & (f <= 12))
                meanAlpha = np.mean(Pxx_den[ind_alpha], axis=0)
                # Beta 12-30
                ind_beta, = np.where((f >= 12) & (f < 30))
                meanBeta = np.mean(Pxx_den[ind_beta], axis=0)
                # Gamma 30-100+
                ind_Gamma, = np.where((f >= 30) & (f < 40))
                meanGamma = np.mean(Pxx_den[ind_Gamma], axis=0)

                feature_vectors[x].insert(y, [meanDelta, meanTheta, meanAlpha, meanBeta, meanGamma])

        powers = np.log10(np.asarray(feature_vectors))

        powers = powers.reshape(-1, 5*2)
        return powers

def svm_model():
    concentrate = pd.read_csv(r'datasets\concentrating.csv', usecols=[1,4])
    normal = pd.read_csv(r'datasets\normal.csv', usecols=[1,4])

    # nfft = nextpow2(256)
    concentrate = concentrate.to_numpy()
    normal = normal.to_numpy()

    data = [concentrate, normal]
    # print(df.shape)

    # plt.scatter(normal[:, 1],concentrate[:, 1], alpha=0.3,
    #             cmap='viridis')
    # plt.show()

    features = []
    for x in data:
        features.append(vectorize(x, 256, filtering=True))

    concentrate_features = features[0]
    normal_features = features[1]

    n_labels = np.full((normal_features.shape[0]), 0)
    c_labels = np.full((concentrate_features.shape[0]), 1)

    n_input = np.column_stack((normal_features, n_labels))
    c_input = np.column_stack((concentrate_features, c_labels))

    full_input = np.concatenate((c_input, n_input))

    # shuffle data
    shuffle_idx = np.random.permutation(len(full_input))
    full_input = full_input[shuffle_idx]


    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(full_input[:, :10], full_input[:, 10], test_size = 0.20)

    from sklearn.svm import SVC
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_train, y_train)
    
    y_pred = svclassifier.predict(X_test)

    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(svclassifier, full_input[:, :10], full_input[:, 10], cv=5)
    print("Mean Score from 5-fold cross-val: "+str(np.mean(scores)))

    return svclassifier

'''
The generation of control signals based on
concentration detection proved that the beta band rises
and theta band declines during concentration and
therefore, the power ratio between beta band and theta
band can be utilized as a parameter in determining a state
of concentration[5]. The index of concentration was used
to determine one’s concentration state based on the
number of occurrences whereby the index of
concentration was higher than the threshold
'''

'''
In addition, according to previous research [23], definite interrelations exist between α and β activities. 
For example, α activity indicates that the brain is in a state of relaxation, whereas β activity is related to stimulation. 
In the study mentioned previously, to observe continuous changes in the mental state of the subjects, the ratio of α and β activities 
was used as the feature for assessing the level of mental attentiveness
'''

'''
In the experiments, the classifiers are support vector machines (SVM), obtained from [29], and multi-layer 
feedforward neural network (BPNN) trained with the back propagation algorithm. SVM has been extensively used 
in many classification problems. Previously, Liu et al. reported that good accuracy (more than 76%) was obtained 
by using SVM to classify two mental states based on EEG [7]. Therefore, we also choose SVM as a classifier. 
To use SVM, one needs to determine many parameters. One key parameter is the kernel type, where a widely used 
one is the RBF (radical basis function) kernel. In the experiments, we also use this type of kernel. 
To use the RBF kernel, we need to provide the value of gamma. In many cases, this parameter is obtained through 
an extensive search. In our case, after some trials, we set this parameter to 8. In addition, we set the cost parameter to 32.
'''
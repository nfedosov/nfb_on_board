#from scipy.io import loadmat

#arr = loadmat('C:/Users/Fedosov/PycharmProjects/online_eeg_prediction/results/online_experiment_01-24_15-00-32/data.mat')

#data = arr['data']


import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.signal as sgn

fs = 512
idc = [0, 2, 21]

filter_len = 51
lag_list = [0,7,12,19,24,31,36,43]
lag_list_ms = (np.array(lag_list)/fs)*1000


for i in range(len(lag_list_ms)):
    if filter_len*1000/fs-lag_list_ms[i] < lag_list_ms[i]:
        lag_list_ms[i] = lag_list_ms[i] -filter_len*1000/fs
num_rep = 5

corr_store = np.zeros((len(lag_list),num_rep))
bias_store = np.zeros((len(lag_list),num_rep))

titles = ['generated signal', 'envelope_on_board', 'received_signal']

b, a = sgn.butter(3, btype='bandpass', Wn=[8, 12], fs=fs)


for count_lag, lag in enumerate(lag_list):
    print(count_lag)
    for rep in range(num_rep):

        path_dir = '_'+str(lag)+'/online_experiment_'+str(rep)+'/data.mat'
        data = loadmat(path_dir)['eeg'][0,:]


        n_matches = 0
        for i in range(3*35):
            if np.array_equal(np.roll(data, shift = -i)[0:6],np.zeros(6)):
                start_idx = i+9
                n_matches += 1
        #print(n_matches)


        data = data[start_idx:]

        data = data[:(len(data)//(35*3))*35*3]


        idc0= np.arange(0,len(data)-2,3)
        idc1 = np.arange(1,len(data)-1,3)
        idc2 = np.arange(2,len(data)-0,3)


        eeg= data[idc0]*0b10000000000000000+data[idc1]*0b100000000+data[idc2]

        eeg = np.reshape(eeg,(-1,35))



        eeg_int21 = eeg[:,21].astype('uint32')
        subtract = np.zeros(1,dtype = 'uint32')
        subtract[0] = 0xFF000000

        idc_of_interest = ((eeg_int21 & 0x00800000)>0) #eeg[:,21] > (0xFFFFFF/2)
        eeg_int21[idc_of_interest] = (~(eeg_int21[idc_of_interest]-1))-subtract


        eeg[idc_of_interest,21] = -(eeg_int21[idc_of_interest]).astype('int32')

        eeg[:,21] *= -1

        eeg /= 0xFFFFFF


        if (rep == 2) and (count_lag == 0):
            for ido, i in enumerate(idc):
                plt.figure()
                plt.plot(eeg[:,i])
                plt.title(titles[ido])



        filtered = sgn.filtfilt(b,a,eeg[:, idc[0]])
        envelope_gt = np.abs(sgn.hilbert(filtered))
        envelope = eeg[:,idc[2]]

        #envelope_gt -= np.min(envelope_gt)
        #envelope -= np.min(envelope)

        envelope_gt -= np.mean(envelope_gt)
        envelope -= np.mean(envelope)

        envelope_gt /= np.linalg.norm(envelope_gt)
        envelope /= np.linalg.norm(envelope)

        Fenvelope_gt =np.fft.fft(envelope_gt).conj()
        Fenvelope = np.fft.fft(envelope)#.conj()

        cc = np.real(np.fft.ifft(Fenvelope_gt*Fenvelope))#/(np.linalg.norm(Fenvelope_gt)*np.linalg.norm(Fenvelope)))

        corr_store[count_lag,rep] = np.max(cc)
        bias_store[count_lag,rep] = np.argmax(cc)*1000.0/fs



mean_bias = np.mean(bias_store,axis = 1)
mean_corr = np.mean(corr_store,axis = 1)

plt.figure()
plt.scatter(np.tile(lag_list_ms,num_rep), (bias_store.T).ravel(), s=10, c = 'blue')
plt.scatter(lag_list_ms, mean_bias, s=20, c = 'red')
plt.xlabel('filter_bias, ms')
plt.ylabel('real_lag, ms')

plt.show()


        #Feeg = np.fft.fft(eeg[:15000,0])
        #Feeg = np.abs(Feeg)
        #plt.figure()
        #plt.plot(Feeg)
        #plt.show()


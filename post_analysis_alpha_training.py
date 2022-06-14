#from scipy.io import loadmat

#arr = loadmat('C:/Users/Fedosov/PycharmProjects/online_eeg_prediction/results/online_experiment_01-24_15-00-32/data.mat')

#data = arr['data']


import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


data = loadmat('/home/cbi/Documents/REALTIME/results/online_experiment_05-17_21-19-30/data.mat')['eeg'][0,:]

for i in range(3*35-6):
    if np.array_equal(data[i:i+6],np.zeros(6)):
        start_idx = i+9

data = data[start_idx:]

data = data[:(len(data)//(35*3))*35*3]


idc0= np.arange(0,len(data)-2,3)
idc1 = np.arange(1,len(data)-1,3)
idc2 = np.arange(2,len(data)-0,3)


eeg= data[idc0]*0b10000000000000000+data[idc1]*0b100000000+data[idc2]

eeg = np.reshape(eeg,(-1,35))


#for i in range(35):
#    plt.figure()
#    plt.plot(eeg[:,i])


fs = 512


idc = [0,2,5]
'''
eeg_int21 = eeg[:,21].astype('uint32')
subtract = np.zeros(1,dtype = 'uint32')
subtract[0] = 0xFF000000

idc_of_interest = eeg[:,21] > (0xFFFFFF/2)
eeg_int21[idc_of_interest] = (~(eeg_int21[idc_of_interest]-1))-subtract


eeg[idc_of_interest,21] = -(eeg_int21[idc_of_interest]).astype('int32')
'''

data = eeg[:,idc[0]]
open = data[:len(data)//2]
closed = data[len(data)//2:]

import scipy.signal as sgn
f, openxx = sgn.welch(open, fs = 512, nperseg =512,noverlap = 512/2)
f, closedxx = sgn.welch(closed, fs = 512, nperseg =512,noverlap = 512/2)

plt.figure()
plt.plot(f[5:30],(openxx[5:30]))
plt.plot(f[5:30],(closedxx[5:30]))

plt.ylim(0,20000)

#eeg[eeg[:,idc[2]]>(0xFFFFFF/2),idc[2]] -= 16777216

#titles = ['generated signal','envelope_on_board','received_signal']

#for ido, i in enumerate(idc):
#    plt.figure()
#    plt.plot(eeg[:,i])
#    plt.title(titles[ido])




#Feeg = np.fft.fft(eeg[:15000,0])
#Feeg = np.abs(Feeg)

#plt.figure()

#plt.plot(Feeg)


b,a = sgn.butter(3,[8,12],btype = 'bandpass', fs = 512)

filtered = sgn.filtfilt(b,a,data)

plt.figure()
plt.plot(filtered)

plt.show()









#id0 = 21
#id1 = 0
#
#bias = np.arange(-20,20,dtype = int)
#
#denom = np.linalg.norm(eeg[:,id0])*np.linalg.norm(eeg[:,id1])
#
# corr = np.zeros(len(bias))
# times = bias*1000/fs
# for i, b in enumerate(bias):
#     corr[i] = np.sum(eeg[:,id0]*np.roll(eeg[:,id1],b))
#
# corr = corr/denom
#
# plt.figure()
# plt.plot(times,corr)
# plt.xlabel('t, ms')
# plt.grid(True)
# plt.show()



'''
uart = serial.Serial('/dev/ttyUSB0')
uart.close()
uart.open()
counter = 0
dlen = 20
data = np.zeros((dlen,35))
uart.reset_input_buffer()
while(counter < dlen):
    if uart.in_waiting>0:
        print(uart.in_waiting)
        counter += 1
        uart.read(uart.in_waiting)
    #if (uart.in_waiting >= 35*3):
    #    uart.read(35)

uart.close()


#uart = UART(1, 9600)                         # init with given baudrate
#uart.init(9600, bits=8, parity=None, stop=1)




from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPainter, QBrush, QPen
from PyQt5.QtCore import Qt
#
# import pylsl
#
# ls = pylsl.resolve_streams()
#
# for stream in ls:
#     print(stream.name())
#

'''
'''
class Square(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        #self.title = "FeedBack Bar"
        self.top = 1000
        self.left = 1000
        self.width = 200
        self.height = 200

        self.rect_top = 0
        self.rect_height = 200
        self.rect_width = 200
        self.rect_left = 0

        self.intensity = 0

        self.InitWindow()

    def InitWindow(self):
        self.setStyleSheet('background-color: #000000;')
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        #self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.setFixedWidth(self.width)
        self.setFixedHeight(self.height)
        self.painter = QPainter(self)
        self.show()


    def paintEvent(self, event):
        self.painter.begin(self)
        #print('PAINT')

        #self.painter.eraseRect(self.rect_left, self.rect_top, self.rect_width, self.rect_height)
        self.painter.setPen(QPen(QtGui.QColor(self.intensity, self.intensity, self.intensity), 1, Qt.PenStyle.SolidLine))
        self.painter.setBrush(QBrush(QtGui.QColor(self.intensity, self.intensity, self.intensity), Qt.BrushStyle.SolidPattern))

        self.painter.drawRect(self.rect_left, self.rect_top, self.rect_width, self.rect_height)


        self.painter.end()

'''
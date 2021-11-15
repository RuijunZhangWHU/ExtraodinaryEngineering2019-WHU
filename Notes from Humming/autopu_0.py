import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
# import pywt
import math

WAVE_LOAD_FILENAME = "Hum.wav"#if rec() is used, = WAVE_OUTPUT_FILENAME
WAVE_OUTPUT_FILENAME = "Record.wav"

def sgn(num):
    '''
        符号函数——未使用
    '''
    if(num > 0.0):
        return 1.0
    elif(num == 0.0):
        return 0.0
    else:
        return -1.0

def enframe(signal, nw, inc):
    '''
    将语音信号转化为帧
    :param signal: 原始信号
    :param nw: 每一帧的长度
    :param inc: 相邻帧的间隔
    :return:
    '''
    signal_length = len(signal)  # 信号总长度
    if signal_length <= nw:      # 如果信号长度小于一帧的长度，则帧数定义为1
        nf = 1                    # nf表示帧数量
    else:
        nf = int(np.ceil((1.0 * signal_length - nw + inc) / inc))  # 处理后，所有帧的数量，不要最后一帧
    pad_length = int((nf - 1) * inc + nw)                          # 所有帧加起来总的平铺后的长度
    pad_signal = np.pad(signal, (0, pad_length - signal_length), 'constant')     # 0填充最后一帧
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf*inc, inc), (nw, 1)).T  # 每帧的索引
    indices = np.array(indices, dtype=np.int32)
    frames = pad_signal[indices]                    # 得到帧信号, 用索引拿数据
    return frames

def autocorr(x, t=1, times=1):
    '''
        calculate autocorrelation funciton
        :param x: data from a frame
        :param t: step
        :param times: lags
        :return:
        '''
    coef = []
    ampdif = []
    div=[]
    for item in range(times):
        temp=np.array([x[0:len(x)-t*(item+1)], x[t*(item+1):len(x)]])
        res=np.corrcoef(temp)
        dif=0.0
        # for i in range(len(x) - t * (item + 1)):
        #     dif = dif+(temp[0,i]-temp[1,i])/6000*(temp[0,i]-temp[1,i])/6000
        # ampdif.append(dif)
        Pcor=res[0,1]*(1-item/len(x)*item/len(x))
        coef.append(Pcor)
        # div.append(Pcor/(dif+1))
    return coef,ampdif,div
def calfreq(test1=[],framerate=16000,wlen=512):
    #去除音频外频率的部分
    for item in range(5):
        test1[item] = 0
    for item in range(50):
        test1[item + wlen - 50 - 1] = 0
    #取最大值\计算频率
    peak = np.argmax(test1)
    freq = framerate / (peak - 1)
    # print(peak)
    # print(freq)
    return freq
def calenergy(frame=[]):
    sum=0
    for i in range(len(frame)):
        sum= sum+frame[i]
    if sum ==0 :
        return 0
    else:
        return 1
def spawmnsoundstep(freqlist=[]):
    soundstep = []
    for i in range(len(freqlist)-2):
        if abs(freqlist[i]-freqlist[i+1])<10 or abs(freqlist[i]-freqlist[i+2])<10:
            soundstep.append(freqlist[i])
        elif abs(freqlist[i+1]-freqlist[i+2])<10:
            soundstep.append(freqlist[i+1])
        else :
            soundstep.append(0)
    soundstep.append(freqlist[len(freqlist)-2])
    soundstep.append(freqlist[len(freqlist) - 1])
    return soundstep
def rec():
    '''
            record sound, saved as directory WAVE_OUTPUT_FILENAME
            '''
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 16000
    RECORD_SECONDS = 5

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    print("Start recording......")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Stop recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wfv = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wfv.setnchannels(CHANNELS)
    wfv.setsampwidth(p.get_sample_size(FORMAT))
    wfv.setframerate(RATE)
    wfv.writeframes(b''.join(frames))
    wfv.close()


# rec() #use when need to record your voice, will be written as directory of WAVE_OUTPUT_FILENAME

# get data and parameters
data = wave.open(WAVE_LOAD_FILENAME,"rb")
params = data.getparams()
nchannels,sampwidth,framerate,nframes = params[:4]
str_data = data.readframes(nframes)
print(framerate)
data.close()
wave_data = np.fromstring(str_data,dtype=np.short)
if nchannels == 2:
    wave_data.shape = -1,2
    wave_data = wave_data.T
    time = np.arange(0,nframes)*(1.0/framerate)
    plt.figure(1)
    plt.subplot(311)
    plt.plot(time,wave_data[0])
    plt.subplot(312)
    plt.plot(time,wave_data[1],c="g")
    # plt.show()
elif nchannels == 1:
    wave_data.shape = -1,1
    wave_data = wave_data.T
    time = np.arange(0,nframes)*(1.0/framerate)
    plt.figure(1)
    plt.subplot(311)
    plt.plot(time,wave_data[0],c="r")
    # plt.show()
data = wave_data[0]
# clipping
Wave_max = max(data)
for i in range(len(data)):
    if abs(data[i])< 0.20*Wave_max:
        data[i]=0
plt.figure(1)
plt.subplot(313)
plt.plot(time,data)
#enframe
wlen = 512
inc = 250
lag = 511
signal = enframe(data,wlen,inc)

#sample
comptime=0.125/4#120pai/s whole=2s 1/16sign=0.125s
distance= 0.125/4/inc*framerate
comps=int(len(signal)/distance)
satimelist=[]
freqlist=[]
for i in range(comps):
    framedata=signal[int(distance)*i,:]
    # print(int(distance*i))
    satime = int(distance)*i * inc / framerate
    satimelist.append(satime)
    print(satime)
    ener = calenergy(framedata)
    if ener==0:
        freqlist.append(0)
    else:
        test1,test2,test3 = autocorr(framedata, 1, lag)
        freq=calfreq(test1,framerate,wlen)#calfreq
        freqlist.append(freq)
plt.figure(1)
plt.subplot(414)
plt.stem(satimelist,freqlist)

SS=spawmnsoundstep(freqlist)
plt.figure(2)
plt.grid(linewidth=0.5)  # 显示网格图
plt.axis([0,satimelist[-1],0,500])
plt.yticks([493,440,391,349,329,293,261,246,220,195,174,164,146,130,123,110,98,87,82,73],["B4","A4","G4","F4","E4","D4","C4","B3","A3","G3","F3","E3","D3","C3","B2","A2","G2","F2","E2","D2"])
# plt.axis([0,satimelist[-1],0,300])
# plt.yticks([293,261,246,220,195,174,164,146,130,123,110,98,87,82,73],["D4","C4","B3","A3","G3","F3","E3","D3","C3","B2","A2","G2","F2","E2","D2"])
plt.plot(satimelist, SS, drawstyle='steps-mid', label='steps-mid')
plt.show()
#get a specific time's correlation function

# framedata = signal[19,:]
# satime=(19-1)*inc/framerate
# print(satime)
# test1,test2,test3=autocorr(framedata,1,lag)
# #去除音频外频率的部分
# for item in range(5):
#     test1[item]=0
# for item in range(50):
#     test1[item+wlen-50-1]=0
# #取最大值\计算频率
# peak = np.argmax(test1)
# freq = framerate/(peak-1)
# print(peak)
# print(freq)
# plt.figure(2)
# plt.subplot(412)
# plt.plot(test3)
# plt.subplot(413)
# plt.plot(test1)
# plt.subplot(414)
# plt.plot(test2)
# plt.show()

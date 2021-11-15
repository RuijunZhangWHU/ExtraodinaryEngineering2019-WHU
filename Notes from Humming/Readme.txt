Hum.Wav, autopu.py, Readme.txt
Author：Ruijun Zhang
Num：2019302120008
Instructor: Fangling Bu
Electronic Information School, Wuhan University

**Hum.wav
A humming audio can be used directly

**autopu.py
The code is used to recognize the notes from a humming audio file. User can either use an existing file in same directory as autopu_0.py such as"Hum.wav" or use the function rec() to record their own audio
Introduction of functions:
enframe：separate long data into frames. Parameters: origin signal,framelenth,frame interval. Return 2 dim frames. User can find any frame by index.
autocorr：calculate correlation function
calfreq：calculate freqency using correlation sequence
spawnsoundstep：generate a step of note that describes the audio
rec：record function

作者：张瑞君
学号：2019302120008
指导老师：卜方玲
武汉大学电子信息学院
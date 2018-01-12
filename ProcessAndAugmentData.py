import numpy as np
from scipy.io import wavfile
from scipy import signal
import librosa

import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import librosa.display

import plotly.offline as py
import plotly.plotly as pg
import plotly.graph_objs as go
import plotly.tools as tls
import pandas as pd
import os,errno

path = r"\Python36\train\audio"
path_original = r"\Python36\Augmented_Data\Original_data"
path_leftshift = r"\Python36\Augmented_Data\LeftShift_data"
path_rightshift = r"\Python36\Augmented_Data\RightShift_data"

path_original_bg = r"\Python36\Augmented_Data\Data_with_bg\Original_bg"
path_leftshift_bg = r"\Python36\Augmented_Data\Data_with_bg\LeftShift_bg"
path_rightshift_bg = r"\Python36\Augmented_Data\Data_with_bg\RightShift_bg"


path_bg = r"\Python36\train\audio\_background_noise_"
bg_files = os.listdir(path_bg)
bg_files.remove('dude_miaowing.wav')
bg_files.remove('README.md')


def createdir(directory):
 try:
     os.makedirs(directory)
 except OSError as e:
     if e.errno != errno.EEXIST:
         raise

noise_rate,noise_samples = wavfile.read('white_noise.wav')
categories =os.listdir(path)
categories.remove('_background_noise_')


count = 0
for speech in categories:
 navigate = path + "\\" + speech
 audiofiles = os.listdir(navigate)
 createdir(path_original + "\\" + speech)
 createdir(path_leftshift + "\\" + speech)
 createdir(path_rightshift + "\\" + speech)
 createdir(path_original_bg + "\\" + speech)
 createdir(path_leftshift_bg + "\\" + speech)
 createdir(path_rightshift_bg + "\\" + speech)
 for audio in audiofiles:
  #plot original data
  sample_rate, samples = wavfile.read(navigate + "\\" + audio)
  S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)
  log_S = librosa.power_to_db(S, ref=np.max)
  fig = plt.figure(frameon=False)
  fig.set_size_inches(0.5,0.5)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
  plt.savefig(path_original + "\\" + speech + "\\" + audio + ".png")
  plt.gcf()
  plt.close()
  plt.clf()
  #
  # plot left shifted wave
  start_ = int(np.random.uniform(300,4800))
  wav_left_shift = np.r_[samples[start_:], np.random.uniform(-0.001,0.001, start_)]
  #
  # now fill the gap
  start_ = np.random.randint(5000,noise_samples.shape[0]-30000)
  bg_slice = noise_samples[start_:start_+len(wav_left_shift)]
  wav_left_shift = wav_left_shift*np.random.uniform(0.95,1.05) + bg_slice*np.random.uniform(0.001,0.002)
  #
  # plot it
  S = librosa.feature.melspectrogram(wav_left_shift, sr=sample_rate, n_mels=128)
  #
  log_S = librosa.power_to_db(S, ref=np.max)
  fig = plt.figure(frameon=False)
  fig.set_size_inches(0.5,0.5)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
  plt.savefig(path_leftshift + "\\" + speech + "\\" + audio + ".png")
  plt.gcf()
  plt.close()
  plt.clf()
  #
  #plot right shifted wave
  start_ = int(np.random.uniform(-4800,-300))
  wav_right_shift = np.r_[np.random.uniform(-0.001,0.001, -start_), samples[:start_]]
  #
  #now fill the gap
  start_ = np.random.randint(5000,noise_samples.shape[0]-30000)
  bg_slice = noise_samples[start_:start_+len(wav_right_shift)]
  wav_right_shift = wav_right_shift*np.random.uniform(0.95,1.05) + bg_slice*np.random.uniform(0.001,0.002)
  #
  # plot it
  S = librosa.feature.melspectrogram(wav_right_shift, sr=sample_rate, n_mels=128)
  #
  log_S = librosa.power_to_db(S, ref=np.max)
  fig = plt.figure(frameon=False)
  fig.set_size_inches(0.5,0.5)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
  plt.savefig(path_rightshift + "\\" + speech + "\\" + audio + ".png")
  plt.gcf()
  plt.close()
  plt.clf()
  #
  #plot original data with noise
  chosen_bg_file = bg_files[np.random.randint(5)]
  bg_rate,bg_samples = wavfile.read(path_bg + "\\" + chosen_bg_file)
  start_ = np.random.randint(5000,bg_samples.shape[0]-30000)
  bg_slice = bg_samples[start_:start_  + len(samples)]
  wav_original_bg = samples*np.random.uniform(0.95, 1.05) + bg_slice*np.random.uniform(0.02, 0.15)
  #
  # plot it
  S = librosa.feature.melspectrogram(wav_original_bg, sr=sample_rate, n_mels=128)
  #
  log_S = librosa.power_to_db(S, ref=np.max)
  fig = plt.figure(frameon=False)
  fig.set_size_inches(0.5,0.5)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
  plt.savefig(path_original_bg + "\\"+ speech + "\\" + audio + ".png")
  plt.gcf()
  plt.close()
  plt.clf()
  #
  # plot left shifted wave with noise 
  chosen_bg_file = bg_files[np.random.randint(5)]
  bg_rate,bg_samples = wavfile.read(path_bg + "\\" + chosen_bg_file)
  start_ = np.random.randint(5000,bg_samples.shape[0]-30000)
  bg_slice = bg_samples[start_:start_  + len(wav_left_shift)]
  wav_left_bg = wav_left_shift*np.random.uniform(0.95, 1.05) + bg_slice*np.random.uniform(0.02, 0.15)
  #
  # plot it
  S = librosa.feature.melspectrogram(wav_left_bg, sr=sample_rate, n_mels=128)
  #
  log_S = librosa.power_to_db(S, ref=np.max)
  fig = plt.figure(frameon=False)
  fig.set_size_inches(0.5,0.5)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
  plt.savefig(path_leftshift_bg + "\\"+ speech + "\\" + audio + ".png")
  plt.gcf()
  plt.close()
  plt.clf()
  #
  #plot right shifted wave with noise
  chosen_bg_file = bg_files[np.random.randint(5)]
  bg_rate,bg_samples = wavfile.read(path_bg + "\\" + chosen_bg_file)
  start_ = np.random.randint(5000,bg_samples.shape[0]-30000)
  bg_slice = bg_samples[start_:start_  + len(wav_right_shift)]
  wav_right_bg = wav_right_shift*np.random.uniform(0.95, 1.05) + bg_slice*np.random.uniform(0.02, 0.15)
  #
  # plot it
  S = librosa.feature.melspectrogram(wav_right_bg, sr=sample_rate, n_mels=128)
  #
  log_S = librosa.power_to_db(S, ref=np.max)
  fig = plt.figure(frameon=False)
  fig.set_size_inches(0.5,0.5)
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
  plt.savefig(path_rightshift_bg + "\\"+ speech + "\\" + audio + ".png")
  plt.gcf()
  plt.close()
  plt.clf()
  count = count + 1
  print("Done with: ",count)
   
  
  
  
  
  



  

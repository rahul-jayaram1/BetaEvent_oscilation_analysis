from bycycle.filt import lowpass_filter
from bycycle.features import compute_features
from bycycle.filt import bandpass_filter
from bycycle.cyclepoints import _fzerorise, _fzerofall, find_extrema

import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import statistics


import os
import scipy.io
import scipy.signal

from scipy import stats


plt.rcParams.update({'font.size': 5})


f_theta = (10,22)
Fs = 600 # sampling rate

filename_pre = 'det_top29pow_pretr_500ms_hitsvmiss_cuttr'
filename_pre = os.path.join('./', filename_pre)
data = sp.io.loadmat(filename_pre)

filename_post = 'det_top29pow_posttr_500ms_hitsvmiss_cuttr'
filename_post = os.path.join('./', filename_post)
data_post = sp.io.loadmat(filename_post)


#srate = data['Fs']

all_signals_hits=data['new_allmeanhits']
all_signals_misses = data['new_allmeanmiss']

all_signals_hits_post = data_post['new_allmeanhits']
all_signals_misses_post = data_post['new_allmeanmiss']


subj = 0

signalsInd_hits = all_signals_hits[subj,:]
signalsInd_misses = all_signals_misses[subj,:]  # change for subject number
signalsInd = np.concatenate([signalsInd_hits, signalsInd_misses])

signalsInd_hits_post = all_signals_hits_post[subj,:]
signalsInd_misses_post = all_signals_misses_post[subj,:]
signalsInd_post = np.concatenate([signalsInd_hits_post, signalsInd_misses_post])

bothSignals_all = np.concatenate([signalsInd, signalsInd_post])




print(type(signalsInd))
signalInd_avg = signalsInd.mean(axis=0)
signal_raw = signalInd_avg #use when plotting fit on avg signal

ind_signal = signalsInd[4] #change to determine indivual trial nunmber

allSubj_avg = []
for i in range(len(all_signals_hits)):
    signalsInd_hits = all_signals_hits[i,:]
    signalsInd_misses = all_signals_misses[i,:]
    sigsInd = np.concatenate([signalsInd_hits, signalsInd_misses])
    sigsInd_avg = sigsInd.mean(axis=0)
    print(len(sigsInd_avg))
    allSubj_avg.append(sigsInd_avg)

print(len(allSubj_avg[0]))

grand_avg = sum(allSubj_avg)/len(allSubj_avg)



signal_raw =ind_signal




# grand_avg = grand_avg/(2*len(all_signals_hits))
#
# for i in range(len(all_signals_hits)):
#     grand_avg = [all_signals_hits[i,:].mean(axis=0), all_signals_misses[i,:].mean(axis=0)])
#     grand_avg += grand_avg
#
# signal_raw = grand_avg

#remove beta event
# print (len(signalsInd))




# for i in range(len(all_signals)):
#     signal_ind = all_signals[i,:].mean(axis=0)
#     sig_avg+=signal_ind
#
# sig_avg_all= sig_avg/len(all_signals)
# signal_raw = sig_avg_all

#signal=signal.transpose() #transpose if time vector and signal are not same dimensions
t = data['new_tVec']

t_raw= t[0]

f_lowpass = 100
N_seconds = 0.4


signalInd_lowpass = lowpass_filter(ind_signal, Fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)

# plt.figure(figsize=(8,6))
# plt.plot(t_raw,ind_signal,'k')
# plt.plot(t_raw, signalInd_lowpass,'b')
# plt.title('Ind signal raw')
# plt.show()


left_edge = 110 #change this value to change length of attached edge signals
right_edge = len(signal_raw) - left_edge


#signal_raw= signal[39,:]

# signal_avg = np.zeros(len(signal.transpose()))
#
# for i in range(len(signal)):
#     signal_avg += signal[i,:]
#
# signal_avg = signal_avg/len(signal)
#
# signal_raw = signal_avg


neg = 0

def signal_addEdges(signal_raw):
    right_edge = len(signal_raw) - left_edge
    if neg == 1:
        sig_left_edge = (-1*signal_raw[left_edge::-1])
        my_signal = np.concatenate([sig_left_edge, signal_raw])
        signal = np.concatenate([my_signal, (-1*signal_raw[:right_edge:-1])])
    elif neg == 0:
        sig_left_edge = (signal_raw[left_edge::-1])
        my_signal = np.concatenate([sig_left_edge, signal_raw])
        signal = np.concatenate([my_signal, (signal_raw[:right_edge:-1])])

    return signal

signal = signal_addEdges(signal_raw)

sig_time = (len(signal_raw)/Fs)

t1=t_raw
t2=t1[right_edge:]-sig_time #subtract time length of signal
t3=t1[:left_edge]+sig_time
t = np.concatenate([t2,t1])
t = np.concatenate([t,t3])
t = t+(sig_time/2)+sig_time
tlim = (0, 3*sig_time) #time interval in seconds
tidx = np.logical_and(t>=tlim[0], t<tlim[1])


signal_low= lowpass_filter(signal, Fs, f_lowpass, N_seconds=N_seconds, remove_edge_artifacts=False)

N_seconds_theta = 0.02
signal_narrow = bandpass_filter(signal, Fs, f_theta, remove_edge_artifacts=False, N_seconds=N_seconds_theta)

Ps, Ts = find_extrema(signal, Fs, f_theta, filter_kwargs={'N_cycles': 3})

tidxPs = Ps[np.logical_and(Ps>tlim[0]*Fs, Ps<tlim[1]*Fs)]
tidxTs = Ts[np.logical_and(Ts>tlim[0]*Fs, Ts<tlim[1]*Fs)]

from bycycle.cyclepoints import find_zerox
zeroxR, zeroxD = find_zerox(signal_low, Ps, Ts)

tidxPs = Ps[np.logical_and(Ps>tlim[0]*Fs, Ps<tlim[1]*Fs)]
tidxTs = Ts[np.logical_and(Ts>tlim[0]*Fs, Ts<tlim[1]*Fs)] #trough time points
tidxDs = zeroxD[np.logical_and(zeroxD>tlim[0]*Fs, zeroxD<tlim[1]*Fs)]
tidxRs = zeroxR[np.logical_and(zeroxR>tlim[0]*Fs, zeroxR<tlim[1]*Fs)]

t_raw_scale = t - (1.5*sig_time)

#
# plt.figure(figsize=(8, 6))
# plt.plot(t[tidx], signal_low[tidx], 'k') #change back to t
# plt.title('Reflected Signal')
# x_coords = [(sig_time/2)+sig_time+t_raw[0], sig_time/2+(2*sig_time)+t_raw[0]]
# for xc in x_coords:
#     plt.axvline(x=xc)
# plt.xlim((tlim))
# plt.show()


plt.figure(figsize=(8, 6))
plt.plot(t[tidx], signal_low[tidx], 'k')
plt.plot(t[tidx],signal_narrow[tidx],'g')
plt.plot(t[tidxPs], signal_low[tidxPs], 'b.', ms=10)
plt.plot(t[tidxTs], signal_low[tidxTs], 'r.', ms=10)
plt.plot(t[tidxDs], signal_low[tidxDs], 'm.', ms=10)
plt.plot(t[tidxRs], signal_low[tidxRs], 'g.', ms=10)
x_coords = [(sig_time/2)+sig_time+t_raw[0], sig_time/2+(2*sig_time)+t_raw[0]]
for xc in x_coords:
    plt.axvline(x=xc)
plt.xlim((tlim))
plt.title('Cycle-by-cycle on Reflected Signal')
plt.show()

from bycycle.features import compute_features
from bycycle.burst import plot_burst_detect_params

burst_kwargs = {'amplitude_fraction_threshold': .18,
                'amplitude_consistency_threshold': .09,
                'period_consistency_threshold': .09,
                'monotonicity_threshold': .1,
                'N_cycles_min': 1}

#burst detect individual trials
# df_ind = compute_features(signal_addEdges(ind_signal), Fs, f_theta,
# center_extrema='T', burst_detection_kwargs=burst_kwargs)
# plot_burst_detect_params(signal_addEdges(ind_signal), Fs, df_ind, burst_kwargs, figsize=(12,1.5))


df = compute_features(signal, Fs, f_theta, center_extrema='T', burst_detection_kwargs=burst_kwargs)
features_mat=df # matrix with all relevant values
#features_mat.to_csv(r'C:\Users\rjayaram\Desktop\Research\cycle-by-cycle\matrix_s10.csv',index = True, header = True)
#df.to_excel('300_300_GrandAvg_peak2peak_HighPower_subj'+'.xlsx')

print('grand avg features')
print(df)

plot_burst_detect_params(signal, Fs, df, burst_kwargs, figsize=(5,4))





# for trial in range(len(signalsInd)):
#     df_ind_loop = compute_features(signal_addEdges(signalsInd[trial]), Fs, f_theta,
#     center_extrema='T', burst_detection_kwargs=burst_kwargs)
#     plot_burst_detect_params(signal_addEdges(signalsInd[trial]), Fs, df_ind_loop, burst_kwargs,figsize=(12,1.5))
#     plt.savefig("subj_"+str(subj)+" trial "+str(trial))

#burst detect indivual subject average
df_ind_avg = compute_features(signal_addEdges(signalInd_avg), Fs, f_theta,
center_extrema='T', burst_detection_kwargs=burst_kwargs)
plt.title('avg signal '+str(subj))
#plot_burst_detect_params(signal_addEdges(signalInd_avg), Fs, df_ind_avg, burst_kwargs, figsize=(12,1.5))








t_raw_scale = t - (1.5*sig_time)


plt.figure(figsize=(8, 6))
# plt.plot(t_raw, sig0, color = '0.75')
# plt.plot(t_raw, sig1, color = '0.75')
# plt.plot(t_raw, sig2, color = '0.75')
# plt.plot(t_raw, sig3, color = '0.75')
# plt.plot(t_raw, sig4, color = '0.75')
# plt.plot(t_raw, sig5, color = '0.75')
# plt.plot(t_raw, sig6, color = '0.75')
# plt.plot(t_raw, sig7, color = '0.75')
# plt.plot(t_raw, sig8, color = '0.75')
# plt.plot(t_raw, sig9, color = '0.75')
plt.plot(t_raw_scale[tidx], signal_low[tidx], 'k')
plt.plot(t_raw_scale[tidxPs], signal_low[tidxPs], 'b.', ms=10)
plt.plot(t_raw_scale[tidxTs], signal_low[tidxTs], 'r.', ms=10)
plt.plot(t_raw_scale[tidxDs], signal_low[tidxDs], 'm.', ms=10)
plt.plot(t_raw_scale[tidxRs], signal_low[tidxRs], 'g.', ms=10)
plt.xlim((t_raw[0], t_raw[-1]))
plt.title('Cycle-by-cycle fitting subj ' + str(subj))
plt.savefig('Cycle-by-cycle fitting pretr subj' + str(subj))
plt.show()

dfs = []
for i in range(len(bothSignals_all)):
    df = compute_features(signal_addEdges(bothSignals_all[i]), Fs, f_theta, center_extrema='T', burst_detection_kwargs=burst_kwargs)
    if i <= int(len(bothSignals_all)/2):
        df['group'] = 'pre'
    else:
        df['group'] = 'post'
    df['trial_number'] = i
    dfs.append(df)
df_cycles = pd.concat(dfs)

print('this is df cycles')
print(df_cycles)

df_cycles_burst = df_cycles[df_cycles['is_burst']]
print(df_cycles['is_burst'])
print(df_cycles_burst)
print('apple')

features_keep = ['period', 'time_rdsym', 'time_ptsym','is_burst']

df_subjects = df_cycles_burst.groupby(['group','trial_number']).mean()[features_keep].reset_index()
print(df_subjects)


feature_names = {'period':'Period (ms)', 'time_rdsym':'Rise-decay symmetry', 'time_ptsym':'Peak-trough symmtery'}

import seaborn as sns
for feat, feat_name in feature_names.items():
    g = sns.catplot( x='group', y=feat, data=df_subjects)
    plt.xlabel('')
    plt.xticks(size=20)
    plt.ylabel(feat_name, size=20)
    plt.yticks(size=15)
    plt.tight_layout()
    plt.show()

#plt.title('pre-beta: lowpass - p/t and r/d midpoints - subject 10')
#plt.title('subject 1 avg p/t and r/d signal')


#plot_burst_detect_params(signal_low, Fs, df, burst_kwargs, tlims=None, figsize=(6,6))

for feat, feat_name in feature_names.items():
    x_treatment = df_subjects[df_subjects['group']=='hits'][feat]
    x_control = df_subjects[df_subjects['group']=='misses'][feat]
    U, p = stats.mannwhitneyu(x_treatment, x_control)
    print('{:20s} difference between groups, U= {:3.0f}, p={:.5f}'.format(feat_name, U, p))

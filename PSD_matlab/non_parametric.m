clc;close all;clear;
%% Read files
% [s, fs, L] = read_audio('../assets/audio/fluteircam.wav'); % s: audio, fs: frame rate   wsubplot(212, 'f (Hz)', '|e^j^\omega|', 'Frequency (Original Signal)'); semilogx(f, P1); 
[s, fs, L] = read_audio('../assets/audio/voiceP.wav');
N = 4000;  segments_time =  (1:N:L-N)/fs; K_bartlett = 4; K_welch= 8;
f0s_bartlett = Bartlett(s, fs, L, N, K_bartlett); % 1.Bartlett
f0s_welch = Welch(s, fs, L, N, K_welch); % 2.Welch

figure;  set(gcf, 'Position', [1300, 600, 600, 600])
ax1_1 = newsubplot(211, 'time (s)', 'f0', 'Bartlett'); h1 = stairs(segments_time, f0s_bartlett); 
h1(1).MarkerFaceColor = 'm'; h1(1).MarkerSize = 4; h1(1).Marker = 'o'; h1(1).LineWidth = 3;
ax1_2 = newsubplot(212, 'time (s)', 'f0', 'Welch'); h2 = stairs(segments_time, f0s_welch); 
h2(1).MarkerFaceColor = 'm'; h2(1).MarkerSize = 4; h1(1).Marker = 'o'; h2(1).LineWidth = 3;


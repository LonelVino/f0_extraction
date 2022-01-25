clc;close all;clear;
%% Read files
% [s, fs, L] = read_audio('../assets/audio/fluteircam.wav'); % s: audio, fs: frame rate   wsubplot(212, 'f (Hz)', '|e^j^\omega|', 'Frequency (Original Signal)'); semilogx(f, P1); 
[s, fs, L] = read_audio('../assets/audio/voiceP.wav');
N = 4000;  segments_time =  (1:N:L-N)/fs;

%% Parametric Methods
order_YW = 16; nfft = 128;
f0_YW = Yule_Walker(s, fs, N, L, order_YW, nfft); %% Yule-Walker AR Method
order_burg = 16; nfft = 128;
f0_burg = Burg(s, fs, N, L, order_burg, nfft); %% Yule-Walker AR Method
p = 8; n_win=2*p; n_overlap = n_win-1; nfft = 1024;
f0_MUSIC = MUSIC(s, fs, N, L, p, nfft, n_win, n_overlap);

%% Plot f0s
figure;  set(gcf, 'Position', [0, 600, 600, 1000])
ax1_1 = newsubplot(311, 'time (s)', 'f0', sprintf('Yule-Walker (order: %d, nFFT: %d)', order_YW, nfft)); h1 = stairs(segments_time, f0_YW); 
h1(1).MarkerFaceColor = 'm'; h1(1).MarkerSize = 4; h1(1).Marker = 'o'; h1(1).LineWidth = 3;
ax1_2 = newsubplot(312, 'time (s)', 'f0', sprintf('Burg (order: %d, nFFT: %d)', order_burg, nfft)); h2 = stairs(segments_time, f0_burg); 
h2(1).MarkerFaceColor = 'm'; h2(1).MarkerSize = 4; h2(1).Marker = 'o'; h2(1).LineWidth = 3;
ax1_3 = newsubplot(313, 'time (s)', 'f0', sprintf('MUSIC (p:%d, win:%d, overlap:%d)', p, n_win, n_overlap)); 
h3 = stairs(segments_time, f0_MUSIC); 
h3(1).MarkerFaceColor = 'm'; h3(1).MarkerSize = 4; h3(1).Marker = 'o'; h3(1).LineWidth = 3;

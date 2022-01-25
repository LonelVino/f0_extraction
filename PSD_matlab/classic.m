clc; close all; clear;

% [s, fs, L] = read_audio('../assets/audio/fluteircam.wav'); % Read files
[s, fs, L] = read_audio('../assets/audio/voiceP.wav');
N = 4000; segments_time =  (1:N:L-N)/fs;
f0s_period = periodogram(s, fs, L, N); % Periodogram: FFT (with Filter and Hamming Window)
f0s_period_ = replace_outliers(f0s_period, 1000);
[f0s_rec, f0s_tri] = AC(s, fs, L, N); % 2.Auto-correlation (Rectangle, Triangle)

figure;  set(gcf, 'Position', [1300, 600, 600, 600])
ax1_1 = newsubplot(311, 'time (s)', 'f0', 'Periodogram'); h1 = stairs(segments_time, f0s_period_); 
h1(1).MarkerFaceColor = 'm'; h1(1).MarkerSize = 4; h1(1).Marker = 'o'; h1(1).LineWidth = 3;
ax1_2 = newsubplot(312, 'time (s)', 'f0', 'Auto-Correlation (Rectangle Window)'); h2 = stairs(segments_time, f0s_rec);
h2(1).MarkerFaceColor = 'm'; h2(1).MarkerSize = 4; h2(1).Marker = 'o'; h2(1).LineWidth = 3;
ax1_3 = newsubplot(313, 'time (s)', 'f0', 'Auto-Correlation (Triangle Window)'); h3 = stairs(segments_time, f0s_tri);
h3(1).MarkerFaceColor = 'm'; h3(1).MarkerSize = 4; h3(1).Marker = 'o'; h3(1).LineWidth = 3;

function f0s = replace_outliers(f0s, threshold)
    B = sort(f0s, 'descend');
    idx = 0;
    for i = B
        idx = idx+1;
        if i < threshold
            break
        end
    end
    mean_f0s = mean(B(idx:end));
    for i = (1:length(f0s))
        if f0s(i) > 1000
            f0s(i) = mean_f0s;
        end
    end    
end
function f0s = periodogram(s, fs, L, N)
    f0s = []; segments = (1:N:L-N);
    for n_start = segments
        x = s(n_start : n_start+N-1); % truncate a segment from original signal, from 10000 to 14000, totally 4000 points
        x1 = filter ([1 -0.97], 1,x); % Pre-add filter, Y=filter(B,A,X), B/A is the coefficient of filter, B is the denominator, A is the numerator
        % Role of filter, remove the drop of 6dB/oct (Decibel / octave band), make the spectrum of the signal flat
        w = (window ('hamming', N)); xw = x1 .* w; % add hamming window, Estimate PSD of the short-time segment
        % apply FFT on time domain  
        Sxw = fft(xw, N);  f_xw = fs*(0:(N/2))/N;  % frequencies of FFT
        % Power Spectrum: Take the square of mode of FFT, divided by signal length into DB
        Sxdb = 20*log10 (abs (Sxw (1 : N/2))) - 10*log10 (N); f_xdb = (0 : N/2-1)*fs / N / 1000;
        if n_start == segments(82)
            figure; set(gcf, 'Position', [600, 500, 700, 1000])
            ax1_1 = newsubplot(311, 'Time (n)', 'Amplitude', 'Intercept Signal (Time Domain)'); plot ((1:length(x))/fs, x); 
            ax1_2 = newsubplot(312, 'Time (n)', 'Amplitude', 'Intercept Signal after adding filter (Time Domain)'); plot ((1:length(x1))/fs, x1);
            ax1_3 = newsubplot(313, 'Time (n)', 'Amplitude', 'Intercept Signal after add hamming window (Time Domain)'); plot ((1:length(xw))/fs, xw);
            figure; set(gcf, 'Position', [1300, 0, 600, 600])
            ax2_1 = newsubplot(211, 'f (Hz)', '|e^j^\omega|', 'Frequency (Intercepted Signal)'); plot(f_xw, abs(Sxw(1:N/2+1))); 
            ax2_2 = newsubplot(212, 'f (kHz)', 'Magnitude (dB)', 'Power Spectrum (Intercepted Signal)'); plot(f_xdb, Sxdb);
        end
        [~, idx_period] = max(Sxdb);  f0_period = f_xdb(idx_period)*1000; 
        f0s(end+1) = f0_period;
    end
end

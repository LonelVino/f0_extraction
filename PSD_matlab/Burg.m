%  pxx: spectral density estimate; f: frequencies cyecles/second (Hz)
%  For real–valued signals, f spans the interval [0,fs/2] when nfft is even and [0,fs/2) when nfft is odd
function f0s = Burg(s, fs, N, L, order, nfft)
    f0s = []; segments = (1:N:L-N); 
    for n_start = segments
        x = s(n_start : n_start+N-1);  
        [PSD, f_fft] = pburg(x,order, nfft, fs);
        [~, idx] = max(PSD); f0 = f_fft(idx);
        f0s(end+1) = f0;
        
        if n_start == segments(81)
            orders = (4:4:16);
            figure;  set(gcf, 'Position', [0, 0, 600, 1200]); set(gca, 'DefaultTextFontSize', 14, 'FontName', 'times')
            set(groot,'defaultAxesFontName','times', 'defaultAxesFontSize', 14)
            idx = 0;
            for order = orders
                idx = idx + 1;
                [pxx, f] = pyulear(x,order, nfft, fs);
                subplot(length(orders),1,idx); 
                plot(f/1000,10*log10(pxx), 'LineWidth', 3);
                title(sprintf('Burg (Order: %d)', order), 'fontsize', 18); 
                xlabel('Frequency (kHz)'); ylabel('Magnitude (dB)');
                grid on;
            end
        end
    end
end
%  pmusic(x,p,nfft,fs,nwin,noverlap)
% p — Subspace dimension
% If p is a real positive integer, then it is treated as the subspace dimension.
% If p is a two-element vector, the second element of p represents a threshold, multiplied by the smallest estimated eigenvalue of the signal's correlation matrix. 
% nwin — Length of rectangular window, 2*p(1) (default) 
% noverlap — Number of overlapped samples, nwin-1 (default) 
function f0s = MUSIC(s, fs, N, L, p, nfft, n_win, n_overlap)
    f0s = []; segments = (1:N:L-N); 
    for n_start = segments
        x = s(n_start : n_start+N-1);  
        [PSD, f_fft] = pmusic(x, p, nfft, fs, n_win, n_overlap);
        [~, idx] = max(PSD); f0 = f_fft(idx);
        f0s(end+1) = f0;
        
        if n_start == segments(81)
            ps = (1:8);
            figure;  set(gcf, 'Position', [1300, 600, 1200, 600]); set(gca, 'DefaultTextFontSize', 14, 'FontName', 'times')
            set(groot,'defaultAxesFontName','times', 'defaultAxesFontSize', 14)
            for p_test= ps
                [pxx, f] = pmusic(x, p_test, nfft, fs);
                subplot(2, length(ps)/2 , p_test); 
                plot(f/1000,10*log10(pxx), 'LineWidth', 3);
                title(sprintf('MUSIC (p: %d)', p_test), 'fontsize', 18); 
                xlabel('Frequency (kHz)'); ylabel('Magnitude (dB)');
                grid on;
            end
        end
    end
end
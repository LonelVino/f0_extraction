function f0s = Welch(s, fs, L, N, K)
    f0s = []; segments = (1:N:L-N); 
    D = fix(N/2 / (K+1)); L = 2*D; % K: the times of overlapping
    for n_start = segments
        Sxw = zeros(1, N/2); 
        w = (window('hamming', L))'; 
        f_xdb = (0 : N/2-1)*fs / N / 1000;
        x = s(n_start : n_start+N-1); 
        for k = 1 : K                %1*8   
            ks = (k-1)*D + 1;       %k=1,ks=1;k=2,ks=223;k=3,ks=445;k=4,ks=667;    k=8,ks=1555    
            ke = ks + L;        %k=1,ke=445;k=2,ke=667                         k=8,ke=1999    
            xk = x (ks:ke)*w; % add hamming window, k=1,444*1 1*444    
            X = (abs(fft(xk, N))).^2;       
            for i = 1 : N/2        
                Sxw (i) = Sxw (i) + X (i); % only keep the first N/2 points, the others are symmetric of the first N/2
            end
        end
        for i = 1 : N/2    
            Sx (i) = 10*log10 (Sxw (i)/(K*L)); % conver into db
        end
        [~, idx] = max(Sxw); f0 = f_xdb(idx)*1000;
        f0s(end+1) = f0; 
        if n_start == segments(81)
            figure; subplot(111);  set(gcf, 'Position', [0, 800, 600, 400])
            plot(f_xdb, Sx); 
            xlabel('Frequency (kHz)', 'FontSize', 14, 'fontname', 'times'); 
            ylabel('Magnitude (dB)', 'FontSize', 14, 'fontname', 'times'); 
            title(sprintf('Welch Estimate, N= (%d), K=(%d), D=(%d), L=(%d)', N, K, D, L), 'FontSize', 18, 'fontname', 'times'); 
        end
    end
end
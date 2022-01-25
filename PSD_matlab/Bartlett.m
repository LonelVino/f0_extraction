function f0s = Bartlett(s, fs, L, N, K)
    f0s = []; segments = (1:N:L-N); L = N/K;
    for n_start = segments
        Sx = zeros(1, N/2); 
        f_xdb = (0 : N/2-1)*fs / N / 1000;
        x = s(n_start : n_start+N-1); 
        for k = 1 : K    
            ks = (k-1)*L + 1;    %k=1,ks=1;  k=2,ks=1001;    
            ke = ks + L - 1;     %k=1,ke=1000 ;k=2,ke=2000;   
            X = fft(x(ks:ke), N);    
            X = (abs (X)).^2;          % 周期图法这里要 abs + 平方 注意    
            for i = 1 : N/2            %i=1:2000        
                Sx(i) = Sx(i) +X(i);             
            end
        end
        for i = 1 : N/2    
            Sx(i) = 10*log10(Sx(i)/(K*L));
        end
        [~, idx] = max(Sx); f0 = f_xdb(idx)*1000;
        f0s(end+1) = f0; 
        if n_start == segments(81)
            figure; subplot(111);  set(gcf, 'Position', [400, 800, 600, 400])
            plot(f_xdb, Sx); 
            xlabel('Frequency (kHz)', 'FontSize', 14, 'fontname', 'times'); 
            ylabel('Magnitude (dB)', 'FontSize', 14, 'fontname', 'times'); 
            title(sprintf('Bartlett Estimate, N=(%d), K=(%d), D=L=(%d)', N, K, L), 'FontSize', 18, 'fontname', 'times'); 
        end
    end
end
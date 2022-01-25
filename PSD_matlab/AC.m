function [f0s_rec, f0s_tri] = AC(s, fs, L, N)
    f0s_rec = []; f0s_tri = []; segments = (1:N:L-N);
    for n_start = segments
        x = s(n_start : n_start+N-1); 
        r = zeros (2*N/2-1, 1); %(-(N/2-1)~(N/2-1)) 
        for k = 1 : N/2               
            x1 = x (k : N);             
            x2 = x (1 : N+1-k);             
            r(N/2+k-1) = x1' * x2 / N;    
            r(N/2-k+1) = r(N/2+k-1);    %r(-k) = r(k)
        end
        f_xdb = (0 : N/2-1)*fs / N / 1000;
        rx_rec = r; Sx_ac_rec = fft(rx_rec, N);   %DFT
        Sxdb_ac_rec = 10*log10(abs(Sx_ac_rec(1 : N/2)));% convert into dB
        [max_db_ac_rec, idx_rec] = max(Sxdb_ac_rec); f0_ac_rec = f_xdb(idx_rec)*1000;

        w = triang (2*N/2-1)'; % add triangle window
        rx_tri = r .* w'; Sx_ac_tri = fft(rx_tri, N);
        Sxdb_ac_tri = 10*log10(abs(Sx_ac_tri(1 : N/2)));
        [max_db_ac_tri, idx_tri] = max(Sxdb_ac_tri); f0_ac_tri = f_xdb(idx_tri)*1000;
        f0s_rec(end+1) = f0_ac_rec; f0s_tri(end+1) = f0_ac_tri;
        if n_start == segments(81)
            figure(4);  set(gcf, 'Position', [1300, 600, 600, 600])
            ax4_1 = newsubplot(211, 'f (kHz)', 'Magnitude (dB)', 'Power Spectrum (AC, Rectangle Window)'); plot(f_xdb, Sxdb_ac_rec); 
            ax4_2 = newsubplot(212, 'f (kHz)', 'Magnitude (dB)', 'Power Spectrum (AC ,Triangle)'); plot(f_xdb, Sxdb_ac_tri);
        end
    end
end
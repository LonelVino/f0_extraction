function [s, fs, L] = read_audio(pth)
    [s, fs] = audioread(pth); % s: audio, fs: frame rate   
    L = length(s); T = 1/fs; % Smaples Length, Sampling period       
    time = linspace(1,L/fs, length(s)); % Time vector
    % Calculate the bilateral spectrum P2.
    Xk=fft (s); P2 = abs(Xk);
    % The single spectrum P1 is then calculated based on the P2 and even signal length.
    P1 = P2(1:L/2+1); P1(2:end-1) = 2*P1(2:end-1);
    f = fs*(0:(L/2))/L;  % frequencies of FFT

    figure;  set(gcf, 'Position', [0,0, 600, 500]);
    ax1_1 = newsubplot(211, 'Time (n)', 'Amplitude', 'Waveform (Time Domain, Original Signal)'); plot (time, s);
    ax1_2 = newsubplot(212, 'f (Hz)', '|e^j^\omega|', 'Frequency (Original Signal)'); semilogx(f, P1); 
end
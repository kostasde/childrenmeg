function [ AcousticSignal, start, stop ] = extractAcousticSignal( subject_ctf )
% Simple function that returns the acoustic signal from a subjects data
% and vectors that indicate the start and stop indices of each trial wrt
% the Acoustic Signal

% Multiple possible places these signals can be found
triggerInd1 = 195;
triggerInd2 = 194;
AcousticSignalInd1 = 194;
AcousticSignalInd2 = 196;

% Some parameter
Threshold = 4.7;
triggerInd = triggerInd1;
AcousticSignalInd = AcousticSignalInd1;

A= subject_ctf.data(:,triggerInd); %The vector with the trigger signal
if mean(A) <= 4.0
    triggerInd = triggerInd2;
    AcousticSignalInd = AcousticSignalInd2;
    A= subject_ctf.data(:,triggerInd);
    if mean(A) <= 4.0
        error('Cannot find the trigger for this subject, handle manually');
        exit(-1);
    end
    Threshold = 4.8;
end

ThreshMatrix = (A>Threshold); %logical values of those bigger than threshold
diffA = diff(ThreshMatrix ); %tries to find when the thresholds were executed
start = find (diffA>0); % this point is the start of the trigger
stop = find(diffA<0); %this point is the end of the trigger

AcousticSignal = subject_ctf.data(:,AcousticSignalInd);

end
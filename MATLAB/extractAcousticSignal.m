function [ AcousticSignal] = extractAcousticSignal( data, fs )
% Simple function that returns the acoustic signal from the two possible
% Acoustic data channels.

% Multiple places these signals can be found, the trigger and audio signal
% should be in one of these two pairs, generally 194 seems to be better 
% audio quality for some reason or another.
triggerInd1 = 195;
triggerInd2 = 194;
AcousticSignalInd1 = 194;
AcousticSignalInd2 = 196;

Threshold = 4.7;
triggerInd = triggerInd1;
AcousticSignalInd = AcousticSignalInd1;

A= data(triggerInd, :);
if mean(A) <= 4.0
    triggerInd = triggerInd2;
    AcousticSignalInd = AcousticSignalInd2;
    A= data(triggerInd, :);
    if mean(A) <= 4.0
        error('Cannot find the trigger for this subject, handle manually');
        exit(-1);
    end
end

%AcousticSignal = zeros(1, size(data,2)*size(data,3));
%for i=1:size(data, 3)
%    AcousticSignal
    
noisefree= ssubmmse(data(AcousticSignalInd, :), fs);
AcousticSignal = squeeze(reshape(noisefree, size(data,2), size(data,3)));

end
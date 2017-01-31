function [ MEGSignal, AudioSignal] = extractSignals( subject_ctf )
% Simple function that returns the MEG signal split up into epochs. Note
% that the epochs will all be the same length, dictated by the SECOND
% largest period of measurement.
% In short this means 

% Some parameters
MEGLowInd = 37;
MEGHighInd = 187;
Threshold = 4.8;
MEGSignal = {};
AudioSignal = {};
j = 1;

% Extract the data signals
MEGSignalContinuous = subject_ctf.data{1, 1}(:, MEGLowInd:MEGHighInd);
[AcousticSignalContinuous, start, stop] = extractAcousticSignal(subject_ctf);

fprintf('%d epochs found...\n', length(start));

% Extract the voiced section for each trial
for i = 1:length(start)
    [segments, fs, limits] = detectVoiced2(AcousticSignalContinuous(...
        start(i):stop(i)), subject_ctf.setup.sample_rate, 0);
    
    % Check if no voicing is detected, if so, warn and skip epoch
    if length(segments) == 0
       fprintf('ERROR: No voice detected!\n'); 
       fprintf('WARNING: Skipping epoch: %d for subject %s\n',...
           i, subject_ctf.setup.subject);
       continue;
    end
    
    % Check if there are multiple, warn and assume it is the first voicing
    if length(segments) > 1
        fprintf('WARNING: multiple voiced sections determined during epoch %d\n', i);
        fprintf('\t...Assuming it is the first section found\n');
    end
    
    % Warn if voiced section is pretty much the entire length of the epoch
    if (limits(1,2) - limits(1,1)) > 0.75*(stop(i) - start(i))
       fprintf('WARNING: Voiced section > 0.75* the length of epoch: %d\n', i); 
       fprintf('... Skipping this, as audio will not be aligned with other epochs\n');
       continue;
    end
    
    % Finally extract
    MEGSignal{j} = MEGSignalContinuous(limits(1,1):limits(1,2), :);
    AudioSignal{j} = segments{1};
    j = j+1;
end

end
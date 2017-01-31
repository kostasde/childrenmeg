function [ MEGdata, AudioData, params ] = preProcMEG(ctf, export_dir, AUDIOONLY)
%PREPROCMEG This function takes the raw MEG signals and applies some
%preprocessing before the feature extraction step:
%       - Downsampling
%[
% Note: there is currently a lot of bloating from copy-pasting, refer to
% above list for convienient interpretation of what this is meant to be
% doing until it gets cleaned up.

addpath('../libs/voicebox_tools');
addpath('../libs/FastICA_25');

MEGIndexLow=37;
MEGIndexHigh=187;
params.fs = ctf.setup.sample_rate;
params.chnames=ctf.sensor.label(MEGIndexLow:MEGIndexHigh);

[MEGdata, AudioData] = extractSignals(ctf);
if isempty(MEGdata)
   fprintf('ERROR: did not extract any good epochs from the data!!!\n\n');
   return;
end
fprintf('\n%d solid epochs found.\n\n', length(MEGdata));

if exist('AUDIOONLY') && AUDIOONLY
    savedata(export_dir, [], AudioData);
    return
end

% Taken from eegExperiments.m
%%%%%%%%%%%%%%%%%%%% Pre-processing params
params.preproc.Steps = {'downsample','laplace', 'ICA', 'normalize'};
params.preproc.Flags = [1 0 0 1];
params.preproc.downsample.freq = 80; % new sampling freq in Hz

% Preprocessing
% Do a downsampling
if (params.preproc.Flags(ismember(params.preproc.Steps,'downsample')) == 1)
  fprintf('Performing a downsampling with new sampling rate = %d \n',params.preproc.downsample.freq);
  Wn=params.preproc.downsample.freq/params.fs;
  MEGdata  = megDownsample(MEGdata, Wn);
  clear Wn
end

% Apply Spatial Filtering - Small Laplacian
if (params.preproc.Flags(ismember(params.preproc.Steps,'spatial')) == 1)
  %fprintf('Applying Spatial Filtering: Small Laplacian . . . \n');
  %MEGdata  = eegSpatialFilterSmallLaplacian(MEGdata, locations);
end

% normalize
if (params.preproc.Flags(ismember(params.preproc.Steps,'normalize')) == 1)
  fprintf('Normalizing . . . \n');
  for i=1:length(MEGdata)
    % First zero
    dc_offset = mean(MEGdata{i});
    MEGdata{i} = bsxfun(@minus, MEGdata{i}, dc_offset);
    % Now Normalize by the deviation
    dev = std(MEGdata{i});
    MEGdata{i} = bsxfun(@rdivide, MEGdata{i}, dev);
  end
end

% Apply ICA
if (params.preproc.Flags(ismember(params.preproc.Steps,'ICA')) == 1)
  fprintf('Applying FastICA. . . \n');
  for i=1:length(MEGdata)
    MEGdata{i}  = fastica(MEGdata{i}', 'verbose', 'off', 'displayMode', 'off')';
    shape = size(MEGdata{i});
    fprintf('%3.2f%%| Found: %d independent components\n', i/length(MEGdata)*100, shape(2));
  end
end

% Added so that we can export all the data as csv to be processed by
% OPENSmile
if exist('export_dir') && ~isempty(MEGdata)
    savedata(export_dir, MEGdata, AudioData);
end

end

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
MEGSignalContinuous = subject_ctf.data(:, MEGLowInd:MEGHighInd);
[AcousticSignalContinuous, start, stop] = extractAcousticSignal(subject_ctf);

% Extract the voiced section for each trial
for i = 1:length(start)
    noisefree = ssubmmse(AcousticSignalContinuous(start(i):stop(i)),...
        subject_ctf.setup.sample_rate);
    [segments, fs, limits] = detectVoiced2(noisefree, subject_ctf.setup.sample_rate, 0);
    
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
    if (limits(1,2) - limits(1,1)) > 0.9*(stop(i) - start(i))
       fprintf('WARNING: Voiced section > 0.9* the length of epoch: %d\n', i); 
       fprintf('... Skipping this, it is very likely not voiced...\n');
       continue;
    end
    
    % Finally extract, we normalize the audio here as well
    MEGSignal{j} = MEGSignalContinuous(limits(1,1):limits(1,2), :);
    AudioSignal{j} = segments{1}/max(segments{1});
    j = j+1;
end
end

function savedata(export_dir, MEGdata, AudioData)
    fprintf('Saving to... %s\n', export_dir);
    if export_dir(end) ~= '/'
        export_dir = [export_dir '/'];
    end
    mkdir(export_dir)
    meg_dir = [export_dir 'MEG/'];
    acoustic_dir = [export_dir 'Acoustic/'];
    
    % Export MEG data if provided
    if ~isempty(MEGdata)
        mkdir(meg_dir)
        for i = 1:length(MEGdata)
            csvwrite(strcat(meg_dir, 'epoch_', int2str(i), '.csv'), MEGdata{i});
        end
    end
    
    % Export audio data if provided
    if ~isempty(AudioData)
        mkdir(acoustic_dir)
        for i = 1:length(AudioData)
            csvwrite(strcat(acoustic_dir, 'epoch_', int2str(i), '.csv'), AudioData{i});
        end
    end
end

function data = megDownsample(data, Wn)

  % Create a 10th order lowpass butterworth filter 
  [b_butr,a_butr] = butter(10,Wn,'low');

  % Run the filter over the data
  for e=1:length(data)
    for c=1:size(data{e}, 2)
      data{e}(:, c) = filter(b_butr, a_butr, data{e}(:, c));
    end
  end
end

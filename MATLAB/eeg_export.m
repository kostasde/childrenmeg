function [ ] = eeg_export( ALLEEG , destination, ind, varargin )
%EEG_EXPORT Export the data from the eeglab format study/set datastructures
% into individual csv files that can be used for the next step in the
% pipeline of actions.
%
% Required Arguments:
%         - ALLEEG: the array of 'sets' that make up a 'study'
%         - destination: The top level directory destination of all the
%                        exported data.
%         - ind: The index of the subject to be exported
%
% Optional Argument Pairs:
%         - 'ICA',          0/1: Disable/Enable the ICA output, or the raw 
%                                signal output of the trials.
%
%         - 'crop',     SECONDS: Enables crop-based data augmentations, 
%                                where given a trial of T seconds, a 
%                                sliding crop t' is produced of SECONDS 
%                                length starting at t'=0, for each sample 
%                                until t'+SECONDS hits the end
%                                of the trial.
%
%         - 'channels, indecies: The indecies of the channels that are to
%                                be extracted (applies to extractchans as
%                                well)
%
%         - 'extractchans', 0/1: True/False export a file that contains the
%                                locations of all channels in use
% 

% Default optional argument parameters
DEFAULT_CROP_SECONDS = 0;
MEGEEG_CHANNELS = 37:187;

% Decode the Optional argument pairs
decoded = finputcheck(varargin, {
    'ICA',           'integer',   [0, 1],  0;
    'crop',          'real',      [],      DEFAULT_CROP_SECONDS;
    'channels',      'integer',   [],      MEGEEG_CHANNELS;
    'extractchans',  'integer',   [0, 1],  0
    });
if isstr(decoded), error('varargin malformatted'); end;


[~, EEG, ~] = pop_newset(ALLEEG, ALLEEG, 1, 'retrieve', ind, 'study', 1);
EEG = eeg_checkset(EEG);
fprintf('Loaded %s\nExporting...\n', EEG.setname);

subj_outdir = fullfile(destination, EEG.subject);
meg_outdir = fullfile(subj_outdir, 'MEG', EEG.condition);
audio_outdir = fullfile(subj_outdir, 'Audio', EEG.condition);

% Export channels file
if decoded.extractchans
   fprintf('Extracting channels file...\n');
   writetable(struct2table(EEG.chanlocs), fullfile(subj_outdir, 'chanlocs.csv'));
end

%   if exist(meg_outdir) == 2 && exist(audio_outdir) == 2, continue; end
mkdir(meg_outdir);
mkdir(audio_outdir);
if size(EEG.data, 3) ~= size(EEG.acoustic, 2)
    warning('%s: Audio and MEG trial numbers must match!', EEG.setname);
end

if decoded.ICA, eegdata = EEG.icaact; else, eegdata = EEG.data; end

fprintf('Epochs: ');

for j=1:size(eegdata, 3)
    if decoded.crop <= 0
        crops = 1:size(eegdata,2);
    else
        croplen = decoded.crop*EEG.srate;
        crops = zeros(size(eegdata,2)-croplen, croplen);
        for i = 0:size(crops, 1)
            crops(i, :) = i:i+croplen;
        end
    end
    
    for i = 1:size(crops, 1)
        ep_fname = strcat('/epoch_', int2str(j)); 
        if exist(fullfile(meg_outdir, strcat(ep_fname, '.bak'))) == 2, continue; end
       
        % transpose for opensmile
        csvwrite(fullfile(meg_outdir, strcat(ep_fname, '.csv')), eegdata(:, crops(i,:),j)');
        csvwrite(fullfile(audio_outdir, strcat(ep_fname, '.csv')), EEG.acoustic(crops(i,:),j));
        fprintf('.');
    end
end
fprintf('\n');

end

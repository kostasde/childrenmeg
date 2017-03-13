% This is the top level script that will output the data that is needed.

% Add path for any libs being used
addpath('../libs/voicebox_tools/');

subjects = {'all'};
toplevel = '/mnt/elephant_sized_space/LizPang/Children';
destination = '/mnt/elephant_sized_space/ALL2';
MEGEEG_CHANNELS = 37:187;

% Create datasets and study to contain them
[STUDY, ALLEEG] = eeglab_process(toplevel, 'subjects', subjects,...
    'destination', destination);
pop_savestudy(STUDY, ALLEEG, 'filepath', STUDY.filepath, 'filename', STUDY.filename);

% If icaact not available calculate, if no ICA weights, perform ICA
if any(cellfun(@isempty, {ALLEEG.icaact})) && ~any(cellfun(@isempty, {ALLEEG.icaweights}))
    std_checkset(STUDY, ALLEEG);
elseif any(cellfun(@isempty, {ALLEEG.icaweights}))
    ALLEEG = pop_runica(ALLEEG, 'icatype', 'runica', 'chanind', ...
        MEGEEG_CHANNELS, 'concatcond', 'on');
    for i=1:length(ALLEEG)
        ALLEEG(i).icachansind = MEGEEG_CHANNELS;
    end
    pop_savestudy(STUDY, ALLEEG, 'filepath', STUDY.filepath, 'filename', STUDY.filename);
end

% Export the datasets
for i=1:length(ALLEEG)
   [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, ALLEEG, 1, 'retrieve', i, 'study', 1);
   EEG = eeg_checkset(EEG);
   fprintf('Loaded %s\nExporting...\n', EEG.setname);
   
   meg_outdir = strcat(EEG.filepath, '/MEG/', EEG.condition, '/');
   audio_outdir = strcat(EEG.filepath, '/Audio/', EEG.condition, '/');
%   if exist(meg_outdir) == 2 && exist(audio_outdir) == 2, continue; end
   mkdir(meg_outdir);
   mkdir(audio_outdir);
   if size(EEG.icaact, 3) ~= size(EEG.acoustic, 2)
       warning('%s: Audio and MEG trial numbers must match!', EEG.setname);
   end
   for j=1:size(EEG.icaact, 3)
       % transpose for opensmile
      csvwrite(strcat(meg_outdir, 'epoch_', int2str(j), '.csv'), EEG.icaact(:,:,j)');
      csvwrite(strcat(audio_outdir, 'epoch_', int2str(j), '.csv'), EEG.acoustic(:,j));
      fprintf('.');
   end
   fprintf('\n');
end


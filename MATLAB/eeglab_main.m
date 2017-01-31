% This is the top level script that will output the data that is needed.

% Put path to eeglab here
addpath('/opt/MATLAB/R2016a/extern/eeglab13_6_5b/');

subjects = {'CC007', 'CC021', 'CC025', 'CC028', 'CC067', 'CC084', 'CC092'};
destination = '/mnt/elephant_sized_space/FEATURES_2';

[STUDY, ALLEEG] = eeglab_process('/mnt/elephant_sized_space/LizPang/Children/',...
    'subjects', subjects, 'destination', destination);
pop_savestudy(STUDY, ALLEEG, 'filepath', STUDY.filepath, 'filename', STUDY.filename);

% Compute icaact and save
%if isempty(ALLEEG.icaact)
%    ALLEEG = pop_runica(ALLEEG, 'icatype', 'runica', 'concatcond', 'on');
%    pop_savestudy(STUDY, ALLEEG, 'filepath', STUDY.filepath, 'filename', STUDY.filename);
%end

% Export the datasets
for i=1:length(ALLEEG)
   [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 1, 'retrieve', i, 'study', 1);
   EEG = eeg_checkset(EEG);
   fprintf('Loaded %s\nExporting...\n', EEG.setname);
   
   outdir = strcat(EEG.filepath, '/MEG/', EEG.condition, '/');
   %if exist(outdir) == 2, continue;
   mkdir(outdir);
   for j=1:size(EEG.icaact, 3)
      csvwrite(strcat(outdir, 'epoch_', int2str(j), '.csv'), EEG.icaact(:,:,j)');
      fprintf('.');
   end
   fprintf('\n');
end


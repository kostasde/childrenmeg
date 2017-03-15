function [ output_args ] = eeg_export( ALLEEG , destination, ind )
%EEG_EXPORT Summary of this function goes here
%   Detailed explanation goes here

[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, ALLEEG, 1, 'retrieve', ind, 'study', 1);
EEG = eeg_checkset(EEG);
fprintf('Loaded %s\nExporting...\n', EEG.setname);

%meg_outdir = strcat(EEG.filepath, '/MEG/', EEG.condition, '/');
%audio_outdir = strcat(EEG.filepath, '/Audio/', EEG.condition, '/');
meg_outdir = fullfile(destination, EEG.subject, 'MEG', EEG.condition);
audio_outdir = fullfile(destination, EEG.subject, 'Audio', EEG.condition);

%   if exist(meg_outdir) == 2 && exist(audio_outdir) == 2, continue; end
mkdir(meg_outdir);
mkdir(audio_outdir);
if size(EEG.data, 3) ~= size(EEG.acoustic, 2)
    warning('%s: Audio and MEG trial numbers must match!', EEG.setname);
end
for j=1:size(EEG.icaact, 3)
    % transpose for opensmile
    f = fullfile(meg_outdir, strcat('/epoch_', int2str(j), '.csv'));
    if exist(fullfile(meg_outdir, strcat('/epoch_', int2str(j), '.bak'))) == 2, continue; end
    csvwrite(f, EEG.icaact(MEGEEG_CHANNELS,:,j)');
    csvwrite(fullfile(audio_outdir, strcat('/epoch_', int2str(j), '.csv')), EEG.acoustic(:,j));
    fprintf('.');
end
fprintf('\n');

end


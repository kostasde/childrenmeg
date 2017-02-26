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
   assert(size(EEG.icaact, 3) == size(EEG.acoustic, 2), 'Audio and MEG trial numbers must match!');
   for j=1:size(EEG.icaact, 3)
       % transpose for opensmile
      csvwrite(strcat(meg_outdir, 'epoch_', int2str(j), '.csv'), EEG.icaact(:,:,j)');
      csvwrite(strcat(audio_outdir, 'epoch_', int2str(j), '.csv'), EEG.acoustic(:,j));
      fprintf('.');
   end
   fprintf('\n');
end


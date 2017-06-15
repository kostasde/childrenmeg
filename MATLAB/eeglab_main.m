% This is the top level script that will output the data that is needed.

% Add path for any libs being used
addpath('../libs/voicebox_tools/');

%subjects = {'all'};
%toplevel = '';
%destination = toplevel;
MEGEEG_CHANNELS = 37:187;

% Export conditions
chanfile = 1;

% Create datasets and study to contain them
[STUDY, ALLEEG] = eeglab_process(toplevel, 'subjects', subjects,...
    'destination', destination);
pop_savestudy(STUDY, ALLEEG, 'filepath', STUDY.filepath, 'filename', STUDY.filename);

%[STUDY, ALLEEG] = pop_loadstudy();

% If icaact not available calculate, if no ICA weights, perform ICA
if any(cellfun(@isempty, {ALLEEG.icaact})) && ~any(cellfun(@isempty, {ALLEEG.icaweights}))
    std_checkset(STUDY, ALLEEG) ;
elseif any(cellfun(@isempty, {ALLEEG.icaweights}))
    for i=1:length(STUDY.condition)
       cond = STUDY.condition(i);
       ind = find(contains({ALLEEG.condition}, cond));
       DAT = [];
       
       ALLEEG = pop_runica(ALLEEG, 'dataset', ind, 'concatenate', 'on', 'chanind', MEGEEG_CHANNELS);
       
       for j = 1:length(ind)
          [ALLEEG, EEG, CURRENSET] = pop_newset(ALLEEG, ALLEEG, 1, 'retrieve', ind(j), 'study', 1);
          EEG = eeg_checkset(EEG);
          eeg_export(ALLEEG, destination, j, 'extractchans', chanfile);
       end       
    end
    pop_savestudy(STUDY, ALLEEG, 'filepath', STUDY.filepath, 'filename', STUDY.filename);
end


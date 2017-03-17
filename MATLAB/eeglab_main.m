% This is the top level script that will output the data that is needed.

% Add path for any libs being used
addpath('../libs/voicebox_tools/');

subjects = {'all'};
toplevel = '/ais/clspace5/spoclab/children_megdata/eeglabdatasets/';
destination = '/ais/clspace5/spoclab/children_megdata/eeglabdatasets/';
MEGEEG_CHANNELS = 37:187;

% Create datasets and study to contain them
%[STUDY, ALLEEG] = eeglab_process(toplevel, 'subjects', subjects,...
%    'destination', destination);
%pop_savestudy(STUDY, ALLEEG, 'filepath', STUDY.filepath, 'filename', STUDY.filename);

[STUDY, ALLEEG] = pop_loadstudy(fullfile(destination, 'smallstudy.study'));

% If icaact not available calculate, if no ICA weights, perform ICA
if any(cellfun(@isempty, {ALLEEG.icaact})) && ~any(cellfun(@isempty, {ALLEEG.icaweights}))
    std_checkset(STUDY, ALLEEG) ;
elseif any(cellfun(@isempty, {ALLEEG.icaweights}))
    for i=3:length(STUDY.condition)
       cond = STUDY.condition(i);
       ind = find(strcmp({ALLEEG.condition}, cond));
       DAT = [];
       fprintf('Found %d tests of %s\n', length(ind), char(cond));
       
       ALLEEG = pop_runica(ALLEEG, 'dataset', ind, 'concatenate', 'on', 'chanind', MEGEEG_CHANNELS);
       
       for j = 1:length(ind)
          [ALLEEG, EEG, CURRENSET] = pop_newset(ALLEEG, ALLEEG, 1, 'retrieve', ind(j), 'study', 1);
          EEG = eeg_checkset(EEG);
          eeg_export(ALLEEG, destination, j);
       end       
    end
    pop_savestudy(STUDY, ALLEEG, 'filepath', STUDY.filepath, 'filename', STUDY.filename);
end


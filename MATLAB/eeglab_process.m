function [ STUDY, ALLEEG ] = eeglab_process( toplevel, varargin )
%EEGLAB_PROCESS Use eeglab to extract MEG signals and perform the following
% preprocessing steps:
%
% - Filter
% - Epoch datasets
% - ICA 
% - Downsample
%
% Usage:
% >> [STUDY, ALLEEG] = eeglab_process(toplevel, key0, val0, ...);
%
% NOTE: eeglab must be added to path to use this function
%
% Optional Key/Value pairs: 
%   'filter' -
% 

% ------------------------------------
% Defualt Parameters for preprocessing
% ------------------------------------
DEFAULT_FILTER = [0.5, 100];
DEFAULT_DOWNSAMPLE = 200;
EP_LABEL_CHOICES = {'Trial_prompt', 'VOnset'};
DEFAULT_EP_LABEL = EP_LABEL_CHOICES{1};
% Seconds window around event above for epoch
DEFAULT_EP_WINDOW = [-0.5, 1.5];

% Experiment types
EXPERIMENT_TYPES = {'PA', 'PDK', 'MO', 'VG'};

% ----------------------------
% Decode key/value arguments
decoded = finputcheck(varargin, {
    'filter',       'real',    [],                   DEFAULT_FILTER;
    'downsample',   'integer', [],                   DEFAULT_DOWNSAMPLE;
    'epoch_label',  'string',  {},                   DEFAULT_EP_LABEL;
    'ica',          'string',  {'none', 'subject', 'group'}, 'none';
    'epoch_window', 'integer', [],                   DEFAULT_EP_WINDOW;
    'subjects',     'cell',    {{}},                 {'all'};
    'destination',  'string',  {},                   toplevel;
    'studyname',    'string',  {},                   'smallstudy'
    }); 

if isstr(decoded), error('varargin malformatted'); end;

% Realistic Values Checks
if length(decoded.filter) ~= 2, decoded.filter = DEFAULT_FILTER; end 
if length(decoded.downsample) ~= 1, decoded.downsample = DEFAULT_DOWNSAMPLE; end 
if length(decoded.epoch_window) ~= 2, decoded.epoch_window = DEFAULT_EP_WINDOW; end 
if decoded.epoch_window(1) >= decoded.epoch_window
   fprintf('Invalid window... Start time after end time\n');
end

% Ensure path has '/' ending
if toplevel(end) ~= '/', toplevel = [toplevel '/']; end
if decoded.destination(end) ~= '/', decoded.destination = [decoded.destination '/']; end
if isempty(strfind(decoded.studyname, '.study')), decoded.studyname = [decoded.studyname '.study']; end

% Check first to see if doing all subjects
if length(decoded.subjects) == 1 && strcmp(decoded.subjects{1}, 'all') == 0
    fprintf(['Preprocessing all subjects in: ', toplevel, '\n']);
    names = dir(toplevel);
    decoded.subjects = {names.name};
end

datasets = {};
wb = waitbar(0, 'Starting Calculation');
% Loop through all our subjects
for i = 1:length(decoded.subjects)   
    if ~isdir([toplevel decoded.subjects{i}])
        fprintf(['Skipping non existent subject: ', decoded.subjects{i}, '...\n']);
        continue
    end
    
    experiments = dir([toplevel decoded.subjects{i} '/*.ds']);
    for j = 1:length(experiments)
        experiment = char(regexp(experiments(j).name, '(?<=[_-]).*(?=([.]))', 'match'));
        wb = waitbar(i/length(decoded.subjects), wb, [decoded.subjects{i} ' : ' experiment]);
        
        % Check if dataset already exists
        if exist([decoded.destination char(decoded.subjects{i}) '/' char(experiment) '.set']) == 2
            fprintf('Skipping existing dataset: %s_%s.set \n', char(decoded.subjects{i}), experiment);
            datasets{length(datasets)+1} = char([decoded.destination decoded.subjects{i} '/' experiment '.set']);
            continue
        end
        
        % Otherwise create and process
        % Read dataset
        try
            EEG = pop_ctf_read([toplevel decoded.subjects{i} '/' experiments(j).name], 'megeeg');
        catch ME
            warning('Failed to read: %s experiment: %s\n', decoded.subjects{i}, experiment);
            continue
        end
        EEG = eeg_checkset(EEG);
        EEG.filepath = char([decoded.destination decoded.subjects{i} '/']);
        if exist([decoded.destination char(decoded.subjects{i}) '/' char(experiment) '.set']) ~= 2
            mkdir(EEG.filepath);
        end
        EEG.filename = strcat(experiment, '.set');
        EEG.subject = decoded.subjects{i};
        EEG.condition = experiment;
        
        % Epoch the data
        EEG = pop_epoch(EEG, {decoded.epoch_label}, decoded.epoch_window);
        EEG = eeg_checkset(EEG);

        % Run an individual ICA if that is specified
        if strcmp(decoded.ica, 'subject')
            EEG = pop_runica(EEG);
        end
        
        % Resample
        EEG = pop_resample(EEG, decoded.downsample);
        EEG = eeg_checkset(EEG);
        
        % Save, and remember path
        pop_saveset(EEG, 'filename', EEG.filename, 'filepath', EEG.filepath);
        fprintf('Saved dataset to: %s/%s\n', EEG.filepath, EEG.filename);
        datasets{length(datasets)+1} = [EEG.filepath EEG.filename];        
    end   
end

if exist([decoded.destination decoded.studyname]) == 2 || ...
        exist([decoded.destination decoded.studyname '.study']) == 2
    [STUDY, ALLEEG] = pop_loadstudy('filepath', decoded.destination, ...
        'filename', decoded.studyname);
    
    % ensure that all the subjects just processed are included in the study
    for i=1:length(datasets)
        found = 0;
        for j=1:length(STUDY.subject)
            if ~isempty(strfind(datasets{i}, STUDY.subject{j}))
                found = 1;
                break
            end
        end
        if ~found
            [STUDY, ALLEEG] = std_editset(STUDY, ALLEEG, 'commands', ...
                {{'index', length(STUDY.subject)+1, 'load', datasets{i}}});
        end
    end
else
    % Create a study out of the datasets
    commands = {};
    for index = 1:length(datasets)
        commands{length(commands)+1} = {'index' index 'load' datasets{index}};
    end
    [STUDY, ALLEEG] = std_editset([], [], 'commands', commands);
    STUDY.filepath = decoded.destination;
    STUDY.filename = decoded.studyname;
end

% eeglab_process  END
end



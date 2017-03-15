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
% NOTE: eeglab must be added to path to use this function, and depending on
% on what is in your path, run eeglab (no need to keep it open, it just adds
% some functions that are used to your path.
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
MEGEEG_CHANNELS = 37:187;

% Experiment types
EXPERIMENT_TYPES = {'PA', 'PDK', 'MO', 'VG'};

% ----------------------------
% Decode key/value arguments
decoded = finputcheck(varargin, {
    'filter',       'real',    [],                   DEFAULT_FILTER;
    'downsample',   'integer', [],                   DEFAULT_DOWNSAMPLE;
    'epoch_label',  'string',  {},                   DEFAULT_EP_LABEL;
    'ica',          'string',  {'none', 'subject', 'group'}, 'none';
    'ica_chan',     'integer', [],                   MEGEEG_CHANNELS;
    'epoch_window', 'integer', [],                   DEFAULT_EP_WINDOW;
    'subjects',     'cell',    {{}},                 {'all'};
    'experiments',  'cell',    {{}},                 EXPERIMENT_TYPES;
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
if length(decoded.subjects) == 1 && strcmp(decoded.subjects{1}, 'all')
    fprintf(['Preprocessing all subjects in: ', toplevel, '\n']);
    names = dir([toplevel 'CC*']);
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
    
    experiments = decoded.experiments;
    for j = 1:length(experiments)
        experiment = experiments{j};
        wb = waitbar(i/length(decoded.subjects), wb, [decoded.subjects{i} ' : ' experiment]);
        
        % Check if dataset already exists
        f = fullfile(decoded.destination, decoded.subjects{i}, strcat(char(experiment), '.set'));
        if exist(f) == 2
            fprintf('Skipping existing dataset: %s_%s.set \n', char(decoded.subjects{i}), experiment);
            datasets{length(datasets)+1} = f;
            continue
        end
        
        % Check if data for experiment and subject exists
        f = fullfile(toplevel, decoded.subjects{i}, ...
            strcat(decoded.subjects{i}, '_', experiment, '.ds'));
        if exist(f) ~= 7, break; end;
    
        % Otherwise create and process
        % Read dataset
        try
            EEG = pop_ctf_read(f, 'all');
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
        
        % Extract Acoustic data and save as new field (MUST BE DONE BEFORE
        % RESAMPLING, DON'T WANT 200Hz Audio!)
        % If fails create error file, have empty entry in datasets
        try
            EEG4000 = pop_epoch(EEG, {decoded.epoch_label}, decoded.epoch_window);
            acousticData = extractAcousticSignal(EEG4000.data, EEG4000.srate);
        catch ME
            warning(ME.message);
            fileID = fopen([EEG.filepath experiment '.error'], 'a+');
            fprintf(fileID, ME.message);
            datasets{length(datasets)+1} = [];
            continue
        end
        EEG.acoustic = acousticData;

        % Run an individual ICA if that is specified
        if strcmp(decoded.ica, 'subject')
            EEG = pop_runica(EEG, 'chanind', decoded.ica_chan);
        end
        EEG.icachansind = decoded.ica_chan;
        
        % Resample
        EEG = pop_resample(EEG, decoded.downsample);
        EEG = eeg_checkset(EEG);
        
        % Epoch the data
        EEG = pop_epoch(EEG, {decoded.epoch_label}, decoded.epoch_window);
        EEG = pop_rmbase(EEG, [decoded.epoch_window(1)*1000 0], []);
        EEG = eeg_checkset(EEG);
        
        try
            % Filtering out EOG signals using BSS methods
            % eigratio - determines the number of principal components that will
            %           be kept in the pre-processing PCA step which is performed before 
            %            any BSS algorithm
            % eog_fd - fractional dimensions as criterion
            % range - specifies the minimum and maximum number of components 
            %         that are to be marked as artifactual in each analysis window.
            EEG = pop_autobsseog(EEG, 440, 440, 'sobi', {'eigratio',1e6}, 'eog_fd', {'range',[2,73]});
        catch ME
            warning(ME.message);
            fileID = fopen([EEG.filepath experiment '.error'], 'a+');
            fprintf(fileID, ME.message);
        end
        
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
        if ~isempty(datasets{index}) 
            commands{length(commands)+1} = {'index' index 'load' datasets{index}};
        end
    end
    [STUDY, ALLEEG] = std_editset([], [], 'commands', commands);
    STUDY.filepath = decoded.destination;
    STUDY.filename = decoded.studyname;
end

% eeglab_process  END
end



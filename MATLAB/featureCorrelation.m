function [ Rho, Pval ] = featureCorrelation( toplevel, varargin )
%FEATURECORRELATION Calculate correlation between audio and meg features
% 
% Key/Value Pairs:
%   splitcond: (default 0) split so that correlation is calculated 
%               independently for each condition available.
%   
%
% NOTE: eeglab must be added to path to use this function, and depending on
% on what is in your path, run eeglab (no need to keep it open, it just adds
% some functions that are used to your path.
%   

% REMOVE THE VESTIGIAL CRAP SOON

% Decode key/value arguments
decoded = finputcheck(varargin, {
    'splitcond',        'boolean',  [0 1],          0,
    'conditions',       'cell',     {{}},  {'MO', 'PA', 'PDK', 'VG'}
    }); 

if isstr(decoded), error('varargin malformatted'); end;

% Subjects table
subjectref = load('/mnt/elephant_sized_space/ALL/subject_table.mat');
subjectref = subjectref.subject_table;

% Main struct that will be used to hold the list of files and tall arrays
% For each condition that correlations will be performed for.
data = struct('all', struct('meg',{},'audio',{}));

% Ensure path has '/' ending
if toplevel(end) ~= '/', toplevel = [toplevel '/']; end

if ~isdir(toplevel)
    fprintf('Directory provided does not exist, exiting...\n');
    exit(-1);
end

subjects = dir([toplevel 'CC*']);
for i=1:length(subjects)
    fprintf('Loading %s...\n', subjects(i).name);
    if ~any(cell2mat(strfind(subjectref{:,'ID'}, subjects(i).name)))
        fprintf('No meta-data for %s, skipping.\n', subjects(i).name);
        continue
    elseif ~isdir([toplevel subjects(i).name '/Audio/'])
        fprintf('No Audio directory, skipping.\n');
        continue
    elseif ~isdir([toplevel subjects(i).name '/MEG/'])
        fprintf('No MEG directory, skipping.\n');
        continue
    end
    
    % Conditions
    acond = dir([toplevel '/' subjects(i).name '/Audio/*']);
    mcond = dir([toplevel '/' subjects(i).name '/MEG/*']);
    if length(acond) ~= length(mcond)
        fprintf('Unequal number of audio and MEG conditions, skipping.\n');
        continue
    end
    for j=1:length(acond)
        if ~any(strcmp(acond(j).name, decoded.conditions))
            fprintf('Found condition %s, not listed in conditions, skipping...\n', acond(j).name);
            continue
        end
        fprintf('%s, ', acond(j).name);       
        aepochs = dir([toplevel subjects(i).name '/Audio/' acond(j).name '/*.mat']);
        mepochs = dir([toplevel subjects(i).name '/MEG/' acond(j).name '/*.mat']);
        
        if isempty(aepochs)
            fprintf('\n');
            continue
        elseif length(aepochs) ~= length(mepochs)
            fprintf('Unequal number of audio and MEG epochs, skipping.\n');
            continue
        end
        % Create real paths for the files
        for k=1:length(mepochs)
            mepochs(k).paths = fullfile(mepochs(k).folder, mepochs(k).name);
        end
        
        if decoded.splitcond, c = acond(j).name; else c = 'all'; end
        if isempty(strmatch(c, fieldnames(data)))
            data = setfield(data, c, struct('meg', [], 'audio', [])); 
            currentdata = getfield(data, c);
            currentdata.meg = {mepochs.paths};
        else
            currentdata = getfield(data, c);
            currentdata.meg = [currentdata.meg mepochs.paths];
        end
                
        % update main datastructure
        data = setfield(data, c, currentdata);
        fprintf('\n');
    end
end % all subjects complete

% Finally actually do the correlations
if decoded.splitcond
    conditions = fieldnames(data);
    Rho = struct();
    Pval = struct();
    for i=2:length(conditions)
        c = getfield(data, char(conditions(i)));
        %fprintf('Checking consistency..\n');
        %checkfiles(c);
        fprintf('Creating Tall matrix for %s...\n', char(conditions(i)));
        fprintf('Mutual Features Correlations\n');
        [r, p] = tallCorr(c, {'MEG'}, {'Audio'});
        Rho = setfield(Rho, char(conditions(i)), r);
        Pval = setfield(Pval, char(conditions(i)), p);
        
        fprintf('Age Correlations\n');
        [r, p] = tallCorr(c, {'MEG', 'Audio'}, {'age'});
        % Silly hack
        r = r(:, 1);
        p = p(:, 1);
        Rho = setfield(Rho, strcat(char(conditions(i)), '_AGE'), r);
        Pval = setfield(Pval, strcat(char(conditions(i)), '_AGE'), p);
        
        save('correlations.mat', 'Rho', 'Pval');
    end
else
    [Rho, Pval] = tallCorr(data.all, 0);
end

end

function [] = checkfiles(st)
% Silly check that needs to be done beacuse datastore reader function cant
% do the error checking...
for i=1:length(st.meg)
    try
        reader(st.meg{i});
        fprintf('%.2f', i/length(st.meg));
    catch E
        fprintf('Inconsistency: %s, removing.\n', st.meg{i});
        st.meg{i} = [];
    end 
end
end

function [r, p] = tallCorr(filestruct, X, Y)

ds = fileDatastore(filestruct.meg, 'ReadFcn', @reader);

t = tall(ds);
%t = t(~cellfun(@isempty, t));

t = cell2underlying(t);

[r, p] = corr(t{:, X}, t{:, Y});

[r, p] = gather(r, p);

end


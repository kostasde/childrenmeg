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

% Decode key/value arguments
decoded = finputcheck(varargin, {
    'splitcond',        'boolean',  [0 1],          0
    }); 

if isstr(decoded), error('varargin malformatted'); end;

data = struct('all', struct('meg',[],'audio',[]));

% Ensure path has '/' ending
if toplevel(end) ~= '/', toplevel = [toplevel '/']; end

if ~isdir(toplevel)
    fprintf('Directory provided does not exist, exiting...\n');
    exit(-1);
end

subjects = dir([toplevel 'CC*']);
for i=1:length(subjects)
    fprintf('Loading %s...\n', subjects(i).name);
    if ~isdir([toplevel subjects(i).name '/Audio/'])
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
        
        if decoded.splitcond, c = acond(j).name; else c = 'all'; end
        if isempty(strmatch(c, fieldnames(data)))
            data = setfield(data, c, struct('meg', [], 'audio', [])); 
        end
        currentdata = getfield(data, c);
        
        % Go through all epoch files
        for k=1:length(aepochs)
            fprintf('.');
            audiodata = load([toplevel subjects(i).name '/Audio/'...
                acond(j).name '/' aepochs(k).name], 'features');
            megdata = load([toplevel subjects(i).name '/MEG/'...
                acond(j).name '/' mepochs(k).name], 'features');
            
            try
                currentdata.meg = [currentdata.meg; megdata.features];
                currentdata.audio = [currentdata.audio; audiodata.features];
            catch ME
                fprintf('\nWARNING: Mismatch in features!!! Not including this\n');
                continue
            end
        end
        fprintf('\n');
        
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
        [r, p] = corr(c.meg, c.audio);
        
        Rho = setfield(Rho, char(conditions(i)), r);
        Pval = setfield(Pval, char(conditions(i)), p);
    end
else
    [Rho, Pval] = corr(data.all.meg, data.all.audio);
end

end


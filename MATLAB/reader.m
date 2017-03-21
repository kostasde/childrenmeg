function [ t ] = reader( filename, varargin )
%READER Summary of this function goes here
%   Detailed explanation goes here  

%MEGLEN = 2869;
%AUDIOLEN = 156;
subjectref = load('/mnt/elephant_sized_space/ALL/subject_table.mat');
subjectref = subjectref.subject_table;

if ~isempty(varargin)
   m = whos(matfile(filename));
   a = whos(matfile(strrep(filename, 'MEG', 'Audio')));
   
   if m(1).size(1) ~= a(1).size(1)
       throw(MException());
   end 
   t = 1;
   return
end

% Load MEG data
megdata = load(filename);
audiodata = load(strrep(filename, 'MEG', 'Audio'));
age = subjectref{regexp(filename, 'CC\d*', 'match'), 'age'};

if size(megdata.features, 1) ~= size(audiodata.features, 1)
    fprintf('Inconsistent files: %s, skipping\n', filename);
    t = {};
    return
end

% Load two columns of age to overcome likely bug in tall implementation of
% corr, where correlation with single dimension ignores dimension
t = table(megdata.features, audiodata.features, ...
    repmat(age, size(megdata.features,1), 2), ...
    'VariableNames', {'MEG','Audio','age'});

end


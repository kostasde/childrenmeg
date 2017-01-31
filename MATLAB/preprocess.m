% This script is used to extract signals that openSMILE can use 
% from the raw MEG data
fprintf('Preprocessing Subject: %d\n', subj_in);

if exist('subj_in') ~= 1
    fprintf('FAILED\nNo subject provided\n');
    exit(-1);
end

if exist('AUDIOONLY') ~= 1
    AUDIOONLY = 0;
end

addpath('../libs/');
rawMEGLocation = '/mnt/elephant_sized_space/LizPang/Children/';
dest = 'temp/';

subject = sprintf('CC%03d/', subj_in);
to_load = strcat(rawMEGLocation, subject);

trials = dir([to_load '*.ds']);
for i = 1:length(trials)
    % Import into matlab format
    ctf = ctf_read(strcat(to_load, trials(i).name));
    % Parse test name
    [GARBAGE, test_name] = strtok(trials(i).name, '_.');
        [test_name, ] = strtok(test_name, '._');
    % Write to temporary directory for analysis
    preProcMEG(ctf, strcat(dest, test_name), AUDIOONLY);
end

fprintf('Complete.\n');
exit(0);

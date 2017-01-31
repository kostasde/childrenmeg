% This script loads all the csv files contained in a named directory, and 
% converts them into a single .mat file to be analyzed

if (exist('in_file') ~= 1) | (exist('out_file') ~= 1)
	fprintf('Need to define in_file and out_file\n');
	exit;
end

fprintf('Loading: %s\n', in_file);
features = importdata(in_file,';',1);
% Transpose temp so that each frame is a contiguous set of rows after
% reshaping the data into a single vector
header = features.colheaders;
features = features.data;%reshape(features.data', [1, numel(features.data)]);
save(out_file, 'features', 'header');
fprintf('Saved to: %s\n', out_file);

exit;

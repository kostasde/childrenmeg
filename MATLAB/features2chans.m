function [ meanf, maxf, minf ] = features2chans( header, features )
%FEATURES2CHANS Takes a set of MEG features and separates location specific
%               activity into matrix that corresponds to channels

separated = cell(151,1);

if length(size(features)) == 2
    features = mean(features, 1, 'omitnan');
end

% If NaN's still exist, set to 0, this is mainly for showing correlation
% and NaN's from correlation are caused by unchanging features, these
% should be removed anyway
features(isnan(features)) = 0;

for i=1:length(separated)
    ind = cellfun(@(x)(~isempty(x)), regexp(header, sprintf('csvdata\\[%d\\]', i-1)));
    
    separated{i} = features(:, ind);
end

biggest = max(cellfun(@length, separated));

w = NaN(biggest, 151);
for i=1:size(w,2)
   w(1:length(separated{i}), i) = separated{i}';
end

meanf = mean(w);
maxf = max(w, [], 2);
minf = min(w, [], 2);

end


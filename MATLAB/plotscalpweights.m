function [  ] = plotscalpweights( w, EEG )
%PLOTSCALPWEIGHTS Plot an interpolated scalp representation of weights 
%
% Usage:
% >> plotscalpweights(w, EEG)
%
% w - input weights, some vector of length 151 to be plotted over the scalp
%
% EEG - A representative EEG subject (it should correspond to the relevant
%       condition being considered

CHANS = EEG.icachansind;

if ~all(size(w) == [1, 151]) && ~all(size(w) == [151, 1])
    fprintf('input weights must be 1 x 151\n');
    return
elseif size(w,2) == 151
    w = w';
end

if isempty(EEG.chanlocs) || isempty(EEG.icawinv) || isempty(EEG.chaninfo)
    fprinf('Invalid EEG');
    return
end

% Sphering is not inversed, should this also be reversed?
w = EEG.icawinv * w;

W = zeros(1,length(EEG.chanlocs));
W(:, CHANS) = w;

topoplot(W, EEG.chanlocs, 'plotchans', CHANS, 'chaninfo', EEG.chaninfo);

end


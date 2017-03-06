function [ xlabels, ylabels ] = correlationFigure( r, p, meglabels, audiolabels, numticks, p_max, plot_title )
%CORRELATIONFIGURE Generate a figure showing a visualiztion of the provided
% matrix
%
% p  - matrix n x m, where m is dimension of meg features, and n the audio
% meglabels - The labels for the meg features
% audiolabels - The labels for the audio features
% numticks - The number of highest value points to label on the axis

if length(size(r)) ~= 2, fprintf('A must be 2 dimensions\n'); return; end
B = abs(r)';

if ~isempty(p)
    % Ignore entries that have p value higher than tolerated
    B(p > p_max) = nan;
end

figure;
get(gca);
ax = gca;
imagesc(B);
colorbar

title(plot_title);
ax.XLabel.String = 'MEG Features';
ax.YLabel.String = 'Acoustic Features';

% Sort the array entries to show the highest activity
B(isnan(B)) = -1;
[~, ind] = sort(B(:), 'descend');

% mark highest activity
[y, x] = ind2sub(size(B), ind(1:numticks));

x = unique(x);
y = unique(y);
    
ax.XTick = x;
ax.YTick = y;

if ~isempty(meglabels) && ~isempty(audiolabels)
    ax.XTickLabel = meglabels(x);
    ax.YTickLabel = audiolabels(y);
    
    xlabels = meglabels(x);
    ylabels = audiolabels(y);
end

end


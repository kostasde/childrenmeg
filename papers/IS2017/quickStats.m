v = 1:5;
A = v' * ones(1,10);
groups = reshape(A',1,50);
vDat = reshape(allDat',1,50);
[p,tbl,stats] = anovan(vDat', {groups'});
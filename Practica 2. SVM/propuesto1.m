#!/usr/bin/octave -qf

load data/hart/tr.dat;
load data/hart/trlabels.dat;
#plot (tr(:,1),tr(:,2),"x")
#plot(tr(trlabels==1,1),tr(trlabels==1,2),"x",tr(trlabels==2,1),tr(trlabels==2,2),"s")
#svmtrain
res = svmtrain(trlabels, tr, '-t 2 -c 1');
#res
ind = res.sv_indices;
coef = res.sv_coef;

ind1 = ind .* (coef>=0);
ind1 = ind1(~all(ind1 == 0, 2),:);
ind2 = ind .* (coef<0);
ind2 = ind2(~all(ind2 == 0, 2),:);

plot(tr(trlabels==1,1),tr(trlabels==1,2),"x",tr(trlabels==2,1),tr(trlabels==2,2),"x",tr(ind1,1),tr(ind1,2),"s",tr(ind2,1),tr(ind2,2),"s")

load data/hart/ts.dat;
load data/hart/tslabels.dat;

svmpredict(tslabels, ts, res, ' ');

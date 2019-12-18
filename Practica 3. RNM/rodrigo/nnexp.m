% 
% Computes the error rate of k nearest neighbors of Y with respect to X
% X  is a n x d training data matrix 
% xl is a n x 1 training label vector 
% Y is a m x d test data matrix
% yl is a m x 1 test label vector 
% epsilons is a row vector with smoothing factors to test
function [err] = nnexp(tr,trlabels,ts,tslabels,nHidden,trPercentage,infoCadaXEpoch,nEpoch,FuncionesActivacionPorCapa)
addpath("./nnet");
mInput = tr';
mOutput=trlabels';
mTestInput=ts';
mTestOutput=tslabels';

%formalizacion de las labels
[validoutDisp,nOutput]=formatear(mOutput);
validoutDisp=validoutDisp';

[nFeat,nSamples]=size(mInput);
nTr=floor(nSamples*trPercentage);
nVal=nSamples-nTr;

rand('seed',23);
indices = randperm(nSamples);



mTrainInput=mInput(:,indices(1:nTr));
mTrainOutput=validoutDisp(:,indices(1:nTr));
mValiInput=mInput(:,indices((nTr+1):nSamples));
mValiOutput=validoutDisp(:,indices((nTr+1):nSamples));





[mTrainInputN,cMeanInput,cStdInput]=prestd(mTrainInput);

VV.P=mValiInput;
VV.T=mValiOutput;

VV.P=trastd(VV.P,cMeanInput,cStdInput);
Pr=minmax(mTrainInputN);
ss=[nHidden,nOutput];
trf=FuncionesActivacionPorCapa;
btf="trainlm";
blf="";
pf="mse";

MLPnet = newff(Pr,ss,trf,btf,blf,pf);

MLPnet.trainParam.show=infoCadaXEpoch;
MLPnet.trainParam.epochs=nEpoch;

net = train(MLPnet,mTrainInputN,mTrainOutput,[],[],VV);

mTestInputN=trastd(mTestInput,cMeanInput,cStdInput);
simOut=sim(net,mTestInputN);

[valor,clasesVotadas] = max(simOut);
err =1- sum(clasesVotadas == mTestOutput)/length(mTestOutput);



endfunction



function [oneHotMatrix,numberOfOutputs] = formatear(matrizInicial)
[nFeat,nSamples]=size(matrizInicial);
mOutputProvisional=sort(unique(matrizInicial));
numberOfOutputs = size(mOutputProvisional,2);
oneHotMatrix=zeros(nSamples,numberOfOutputs);
for i=[1:nSamples]
  oneHotMatrix(i,matrizInicial(i))=1;
endfor

endfunction

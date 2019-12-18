% Computes the error rate of k nearest neighbors of Y with respect to X
% X  is a n x d training data matrix 
% xl is a n x 1 training label vector 
% Y is a m x d test data matrix
% yl is a m x 1 test label vector 
% epsilons is a row vector with smoothing factors to test
function multinomial(X,xl,Y,yl,epsilons)

classes=unique(xl);
N=rows(X);
M=rows(Y);
D=columns(X);

pc=[];
pxGc=[];
for c=classes'
  idx=find(xl==c);
  pc=[pc rows(idx)/N];
  cxGc=sum(X(idx,:))';
  pxGc=[pxGc cxGc/sum(cxGc)];
end

printf("\n    eps tr-err te-err");
printf("\n------- ------ ------\n");

% IMPORTANT epsilons as a row vector
for e=epsilons

% Compute Laplace smoothing
spxGc=(pxGc+e)/(1.0+e*D);

% Compute g for each sample in the training set
g=ones(N,1)*log(pc)+X*log(spxGc);
[~,idx]=max(g');
trerr=mean(classes(idx)!=xl)*100;

% Compute g for each sample in the test set
g=ones(M,1)*log(pc)+Y*log(spxGc);
[~,idy]=max(g');
teerr=mean(classes(idy)!=yl)*100;

printf("%.1e %6.3f %6.3f\n",e,trerr,teerr);

end

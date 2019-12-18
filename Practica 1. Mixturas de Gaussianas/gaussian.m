% Computes the error rate of k nearest neighbors of Y with respect to X
% X  is a n x d training data matrix 
% xl is a n x 1 training label vector 
% Y is a m x d test data matrix
% yl is a m x 1 test label vector 
% alpha is the smoothing weight of the full covariance matrix
function [teerr] = gaussian(X,xl,Y,yl,alpha)

classes=unique(xl);
N=rows(X);
M=rows(Y);
D=columns(X);

% Parameter estimation
for c=classes'
  ic=find(c==classes);
  idx=find(xl==c);
  Xc=X(idx,:);
  Nc=rows(Xc);
  pc(ic)=Nc/N;
  muc=sum(Xc)/Nc;
  mu(:,ic)=muc';
  sigma{ic}=((Xc-muc)'*(Xc-muc))/Nc;
  % Smoothing with identity matrix
  sigma{ic}=alpha*sigma{ic}+(1-alpha)*eye(D);
end

% Compute g for each sample in the training set
%for c=classes'
%  ic=find(c==classes);
%  gtr(:,ic)=log(pc(ic))+compute_pxGc(mu(:,ic),sigma{ic},X);
%end
%
%[~,idx]=max(gtr');
%trerr=mean(classes(idx)!=xl)*100;

% Compute g for each sample in the test set
for c=classes'
  ic=find(c==classes);
  gte(:,ic)=log(pc(ic))+compute_pxGc(mu(:,ic),sigma{ic},Y);
end

[~,idy]=max(gte');
teerr=mean(classes(idy)!=yl)*100;

end

% Computes component-and-class conditional gaussian prob
function [pxGc] = compute_pxGc(mu,sigma,X)
  qua=-0.5*sum((X*pinv(sigma)).*X,2);
  lin=X*(mu'*pinv(sigma))';
  cons=-0.5*logdet(sigma);
  cons=cons-0.5*mu'*pinv(sigma)*mu;
  pxGc=qua+lin+cons;
end

% Robust computation of the logarithm of the determinant of the covariance matrix X
% https://www.adelaide.edu.au/mathslearning/play/seminars/evalue-magic-tricks-handout.pdf
% The det of X is the product of its eigenvalues, 
% then the log of the det is the sum of the log of the eigenvalues
% If there are eigenvalues that are zero (or negative), 
% then the log of the det is the smallest value
function v = logdet(X)
  lambda = eig(X);
  if any(lambda<=0)
    v=log(eps);
  else
    v=sum(log(lambda));
  end
end

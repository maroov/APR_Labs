#!/snap/bin/octave -qf


%usamosPca si pcaMin no es -1
if (nargin!=12)
printf("Usage: nnet-exp.m <tr>,trlabels,ts,tslabels,nHidden,trPercentage,infoCadaXEpoch,nEpoch,pcaMin,pcaPaso,pcaMax, soyMnist\n")
exit(1);
end;

arg_list=argv();
trdata=arg_list{1};
trlabs=arg_list{2};
tedata=arg_list{3};
telabs=arg_list{4};
nHidden=str2num(arg_list{5});
trPercentage=str2num(arg_list{6});
infoCadaXEpoch=str2num(arg_list{7});
nEpoch=str2num(arg_list{8});
pcaMin=str2num(arg_list{9});
pcaPaso=str2num(arg_list{10});
pcaMax=str2num(arg_list{11});
soyMnist=str2num(arg_list{12});
FuncionesActivacionPorCapa={"tansig","logsig"};

#Cargamos las matrices
load(trdata);
load(trlabs);
load(tedata);
load(telabs);

if(soyMnist==1)
    tr=X;
    trlabels=xl;
    trlabels=trlabels+1;
    ts=Y;
    tslabels=yl;
    tslabels=tslabels+1;
endif

printf("Numero de neuronas usadas %d, porcentaje entrenamiento: %f, con estas funciones de activacion:\n",nHidden,trPercentage);
disp(FuncionesActivacionPorCapa);
%no usamos pca
if pcaMin== -1
    err=nnexp(tr,trlabels,ts,tslabels,nHidden,trPercentage,infoCadaXEpoch,nEpoch,FuncionesActivacionPorCapa);
    printf("Error\t%f\n",err);
else
    [trPCAmean, trPCAMatrix]=pca(tr);
    printf("PCA\tError\n");
    for c = pcaMin:pcaPaso:pcaMax
        trAct = trPCAMatrix(:,1:c)'*(tr -trPCAmean)';
        teAct = trPCAMatrix(:,1:c)'*(ts -trPCAmean)';
        error = nnexp(trAct',trlabels,teAct',tslabels,nHidden,trPercentage,infoCadaXEpoch,nEpoch,FuncionesActivacionPorCapa)
        printf("%d\t%f\n",c,error);
    endfor
endif


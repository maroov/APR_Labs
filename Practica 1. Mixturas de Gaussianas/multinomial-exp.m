#!/usr/bin/octave -qf

addpath("../../src/multinomial/");

if (nargin!=5)
printf("Usage: multinomial-exp.m <trdata> <trlabels> <tedata> <telabels> <epsilons>\n")
exit(1);
end;

arg_list=argv();
trdata=arg_list{1};
trlabs=arg_list{2};
tedata=arg_list{3};
telabs=arg_list{4};
epsilons=str2num(arg_list{5});

#Cargamos las matrices
load(trdata);
load(trlabs);
load(tedata);
load(telabs);

% Samples and labels are expected to come as rows
multinomial(X,xl,Y,yl,epsilons);

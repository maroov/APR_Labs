#!/usr/bin/octave -qf

if (nargin!=4)
	printf( "Usage: mixgaussian-exp.m <trdata> <trlabels> <tedata> <telabels>\n" );
	exit(1);
end;

arg_list = argv();
trdata = arg_list{ 1 };
trlabs = arg_list{ 2 };
tedata = arg_list{ 3 };
telabs = arg_list{ 4 };

load( trdata );
load( trlabs );
load( tedata );
load( telabs );

[ avg reducedMat ] = pca( X );

save avg.mat avg
save reducedMat.mat reducedMat
save trdata 
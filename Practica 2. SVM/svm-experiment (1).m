#!/usr/bin/octave
load( "data/hart/tr.dat" );
load( "data/hart/trlabels.dat" );
load( "data/hart/ts.dat" );
load( "data/hart/tslabels.dat" );
res = svmtrain( trlabels, tr, '-t 2 -c 1' );
# plot( tr( intersect( res.sv_indices, find( trlabels ) == 1), 1),
#       tr( intersect( res.sv_indices, find( trlabels ) == 1), 2),"x",
#       tr( intersect( res.sv_indices, find( trlabels ) == 2), 1),
#       tr( intersect( res.sv_indices, find( trlabels ) == 2), 2),"s" )

svmpredict( tslabels, ts, res, ' ' );
# TODO: calcular intervalos confianza




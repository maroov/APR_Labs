#!/usr/bin/octave -qf

if (nargin!=5)
	printf( "Usage: gaussian-exp.m <trdata> <trlabels> <tedata> <telabels> <epsilons>\n" );
	exit(1);
end;

arg_list = argv();
trdata = arg_list{ 1 };
trlabs = arg_list{ 2 };
tedata = arg_list{ 3 };
telabs = arg_list{ 4 };
epsilons = arg_list{ 5 };

load( trdata );
load( trlabs );
load( tedata );
load( telabs );

[ avg reducedMat ] = pca( X );
size( reducedMat );

val = [];
min_error_dimensions = 0;
min_error = 100;

mat = zeros(10,columns(epsilons));

for j = 1 : 1 : columns(epsilons)
  for i = 10 : 10 : 100
    reducedMatTraining = (reducedMat( :, 1:i )' * (X - avg)')';
    reducedMatTesting  = (reducedMat( :, 1:i )' * (Y - avg)')';
    error_gaussian = gaussian( reducedMatTraining, xl, reducedMatTesting, yl, epsilons(j) );
    if( min_error > error_gaussian )
      min_error = error_gaussian;
      min_error_dimensions = i;
    endif
    val = [ val; error_gaussian ];
  endfor
  mat(:,j) = val;
  val = [];
endfor

#0.1,0.2,0.5,0.9,0.95,0.99,1
printf(columns(epsilons))
for i=1:columns(epsilons)
  printf("%f\t%f\n",epsilons(i),mat(:,i)');
endfor

#printf( "The dimensions of minimum error are %d\n\n, with error = %f\n\n", min_error_dimensions, min_error );

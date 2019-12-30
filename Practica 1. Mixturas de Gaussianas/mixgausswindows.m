#!/usr/bin/octave -qf
disp("Cargando datos...")
load("data/train-images-idx3-ubyte.mat.gz");
load("data/train-labels-idx1-ubyte.mat.gz");
load("data/t10k-images-idx3-ubyte.mat.gz");
load("data/t10k-labels-idx1-ubyte.mat.gz");
load avg.mat
load reducedMat.mat
disp("Datos cargados.")

min_error_dimensions = 0;
min_error = 100;

for j = [1]
  for i = 10 : 10 : 100
    for k = [1 2 3 4 5 6 7 8 9 10]
      printf("%d dimensiones PCA, K = %d y alfa = %.2f\n", i, k, j);
      reducedMatTraining = (reducedMat( :, 1:i )' * (X - avg)')';
      reducedMatTesting  = (reducedMat( :, 1:i )' * (Y - avg)')';
      error_gaussian = mixgaussian( reducedMatTraining, xl, reducedMatTesting, yl, k, j );
      if( min_error > error_gaussian )
        min_error = error_gaussian;
        min_error_dimensions = i;
      endif
      printf("Error: %.2f\n", error_gaussian);
    endfor
  endfor

endfor

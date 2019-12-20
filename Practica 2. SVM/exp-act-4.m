#!/usr/bin/octave

# Carga de datos
printf("Loading data...\n");
load("data/MNIST/train-images-idx3-ubyte.mat.gz");
load("data/MNIST/train-labels-idx1-ubyte.mat.gz");
load("data/MNIST/t10k-images-idx3-ubyte.mat.gz");
load("data/MNIST/t10k-labels-idx1-ubyte.mat.gz");
printf("Finished loading.\n");

kernel = 1; #[0 1 2 3]

C = 0.1; #[0.001 0.1 1 10 1000]

degree = 2; #[ 1? 2 3 4 5 6 7 8 9 ]

# Entrenamiento
printf("\nTraining...");
res = svmtrain( xl, X, ["-t ", num2str(kernel), " -d ", num2str(degree)," -c ", num2str(C)] );

# Clasificacion
printf("\nClassifying...")
test = svmpredict( yl, Y, res, "" );

# Error
numSamples = rows(yl);
numErr = sum(test != yl);
error = 100 * numErr / numSamples;

# Output
printf("\nKernel = %d\tC = %d\t Error: %.2f\n\n", kernel, C, error);

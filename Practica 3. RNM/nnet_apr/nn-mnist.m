#!/usr/bin/octave

addpath("nnet");

k = 5;
numNeuronsHiddenLayer = 10;


printf( "Loading files...\n" );
# X
load( "data/MNIST/train-images-idx3-ubyte.mat.gz" );
# xl
load( "data/MNIST/train-labels-idx1-ubyte.mat.gz" );

# Y
load( "data/MNIST/t10k-images-idx3-ubyte.mat.gz" );
# yl
load( "data/MNIST/t10k-labels-idx1-ubyte.mat.gz" );
printf( "Finished loading files.\n" );

printf( "Starting PCA...\n" );
# Reduce dimensionality
[ average W ] = pca( X );
printf( "PCA ended.\n" );

trainInput = W(:, 1:k)' * (X - average)';
# trainInput = trainInput';
trainOutput = xl';
testInput = W(:, 1:k)' * (Y - average)';
# testInput = testInput';
testOutput = yl';


[ numFeatures, numSamplesTraining ] = size( trainInput );
# 50% of the training samples are dedicated to real training
numTraining = floor( numSamplesTraining * 0.5 );
numValidation = numSamplesTraining - numTraining;

rand( "seed", 23 );
indices = randperm( numSamplesTraining );

samplesTrainInput = trainInput( :, indices( 1 : numTraining ) );
samplesTrainOutput = trainOutput( indices( 1 : numTraining ) );
samplesValidationInput = trainInput( :, indices( numTraining + 1 : numSamplesTraining ) );
samplesValidationOutput = trainOutput( indices( numTraining + 1 : numSamplesTraining ) );
numClasses = max( columns( unique( trainOutput ) ), columns( unique( testOutput ) ) );

# Class 1 = [ 1, 0, ..., 0 ]
# Class 2 = [ 0, 1, ..., 0 ]
# vector[ i ] = 1 if i = class, 0 otherwise
# TODO: make this a function
oneHotTraining = zeros( numClasses, numTraining );
for class = 1 : numClasses
    oneHotTraining( class, find( samplesTrainOutput == class ) ) = 1;
endfor

oneHotValidation = zeros( numClasses, numValidation );
for class = 1 : numClasses
    oneHotValidation( class, find( samplesValidationOutput == class ) ) = 1;
endfor

# Normalize samples
[ trainInputNormalized, meanInput, stdInput ] = prestd( samplesTrainInput );

# Validation requires a special structure
VV.P = samplesValidationInput;
VV.T = oneHotValidation;
VV.P = trastd( VV.P, meanInput, stdInput );

# Get the parameters to train the neural network
# Minimum and maximum values for each dimension
minMaxValues = minmax( trainInputNormalized );
# Row vector with the number of neurons for each layer
numNeurons = [ numNeuronsHiddenLayer, numClasses ];
# Activation functions for each layer
activationFunctions = { "tansig", "logsig" };
# Algorithm to train the neural network (backpropagation)
trainingAlgorithm = "trainlm";
# Some weird parameter
blf = "";
# Objective function to minnumClassesmize (minimum square error)
objectiveFunction = "mse";
# Train the neural network
nnConfig = newff( minMaxValues, numNeurons, activationFunctions, trainingAlgorithm, blf, objectiveFunction );

# Set training parameters
nnConfig.trainParam.show = 10;
nnConfig.trainParam.epochs = 300;

nn = train( nnConfig, trainInputNormalized, oneHotTraining, [], [], VV );

testInputNormalized = trastd( testInput, meanInput, stdInput );
output = sim( nn, testInputNormalized );


maximum = max( output );
results = zeros();
# For each class, check if the maximum probability of the samples
# to be classified in a class coincides with the class probability
for class = 1 : numClasses
    classifiedIndices = find( output( class, : ) == maximum );
    results( classifiedIndices ) = class;
endfor

error = sum( results != testOutput ) / columns( testOutput ) * 100;
error

# error = sum(results != testlabels)/rows(testlabels)
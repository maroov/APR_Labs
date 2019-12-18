#!/usr/bin/octave

C = 1000

# Conjunto linealmente separable
load( "data/mini/trSep.dat" );
load( "data/mini/trSeplabels.dat" );
trSep = tr;
trSepLabels = trlabels;

# Conjunto no linealmente separable
load( "data/mini/tr.dat" );
load( "data/mini/trlabels.dat" );
trNotSep = tr;
trNotSepLabels = trlabels;

resSep = svmtrain( trSepLabels, trSep, "-t 0 -c 1000" );
resNotSep = svmtrain( trNotSepLabels, trNotSep, "-t 0 -c 1000" );

# Multiplicadores de Lagrange
multsSep = resSep.sv_coef
multsNotSep = resNotSep.sv_coef;


# Índices de los vectores soporte
indSep = resSep.sv_indices;
indNotSep = resNotSep.sv_indices;


# Class 1 -> 1
# Class 2 -> -1
classSep = trSepLabels( indSep );
classSep( find( classSep == 2 ) ) = -1;

classNotSep = trNotSepLabels( indNotSep );
classNotSep( find( classNotSep == 2 ) ) = -1;


# Weights are given as a row vector
weightsSep = sum( multsSep .* trSep( indSep, : ) )
firstValid = indSep( find( multsSep < C ) )( 1 );
thresholdSep = classSep( firstValid ) - weightsSep * trSep( firstValid, : )'

weightsNotSep = sum( multsNotSep .* trNotSep( indNotSep, : ) )
firstValid = indNotSep( find( multsNotSep < C ) )( 1 );
thresholdNotSep = classNotSep( firstValid ) - weightsNotSep * trNotSep( firstValid, : )'
toleranceMarginsNotSep = 1 .- classNotSep .* (weightsNotSep * trNotSep(indNotSep, :)' .+ thresholdNotSep)'

# Margin
marginSep = 2 / norm( weightsSep )
marginNotSep = 2 / norm( weightsNotSep )


# Frontier parameters
slopeSep = -weightsSep( 1 ) / weightsSep( 2 );
interceptSep = -thresholdSep / weightsSep( 2 );
interceptSepM1 = -(thresholdSep - 1) / weightsSep( 2 );
interceptSepM2 = -(thresholdSep + 1) / weightsSep( 2 );

slopeNotSep = -weightsNotSep( 1 ) / weightsNotSep( 2 );
interceptNotSep = -thresholdNotSep / weightsNotSep( 2 );
interceptNotSepM1 = -(thresholdNotSep - 1) / weightsNotSep( 2 );
interceptNotSepM2 = -(thresholdNotSep + 1) / weightsNotSep( 2 );

# Plot
plotIndSepFirstClass = intersect( indSep, find( trSepLabels == 1 ) );
plotIndSepSecondClass = intersect( indSep, find( trSepLabels == 2 ) );

plotIndNotSepFirstClass = intersect( indNotSep, find( trNotSepLabels == 1 ) );
plotIndNotSepSecondClass = intersect( indNotSep, find( trNotSepLabels == 2 ) );

figure( 1 );
plot(
        # Support vectors
        trSep( plotIndSepFirstClass, 1 ), trSep( plotIndSepFirstClass, 2 ), "rx",
        trSep( plotIndSepSecondClass, 1 ), trSep( plotIndSepSecondClass, 2 ), "bx",
        # All samples
        trSep( trSepLabels == 1, 1 ), trSep( trSepLabels == 1, 2 ), "ro",
        trSep( trSepLabels == 2, 1 ), trSep( trSepLabels == 2, 2 ), "bo"
    );
line( "xdata", [ 0, 7 ], "ydata", [ interceptSep, 7 * slopeSep + interceptSep ], "linewidth", 3, "color", "k" )
line( "xdata", [ 0, 7 ], "ydata", [ interceptSepM1, 7 * slopeSep + interceptSepM1 ], "linewidth", 3, "color", "m" )
line( "xdata", [ 0, 7 ], "ydata", [ interceptSepM2, 7 * slopeSep + interceptSepM2 ], "linewidth", 3, "color", "c" )
figure( 2 );
plot(
        # Support vectors
        trNotSep( plotIndNotSepFirstClass, 1 ), trNotSep( plotIndNotSepFirstClass, 2 ), "rx",
        trNotSep( plotIndNotSepSecondClass, 1 ), trNotSep( plotIndNotSepSecondClass, 2 ), "bx",
        # All samples
        trNotSep( trNotSepLabels == 1, 1 ), trNotSep( trNotSepLabels == 1, 2 ), "ro",
        trNotSep( trNotSepLabels == 2, 1 ), trNotSep( trNotSepLabels == 2, 2 ), "bo"
    );
line( "xdata", [ 0, 7 ], "ydata", [ interceptNotSep, 7 * slopeNotSep + interceptNotSep ], "linewidth", 3 )
line( "xdata", [ 0, 7 ], "ydata", [ interceptNotSepM1, 7 * slopeNotSep + interceptNotSepM1 ], "linewidth", 3, "color", "m" )
line( "xdata", [ 0, 7 ], "ydata", [ interceptNotSepM2, 7 * slopeNotSep + interceptNotSepM2 ], "linewidth", 3, "color", "c" )
axis( [ 0 7 0 7 ] )
pause
#fplot( trSep( trSepLabels == 1, 1 ), trSep( trSepLabels == 1, 2 ), "x" );
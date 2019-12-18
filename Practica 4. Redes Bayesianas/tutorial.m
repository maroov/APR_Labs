%% APR - Práctica 4: Redes Bayesianas
%% Nahuel Unai Roselló Beneitez
%% Manuel Roselló Oviedo

%% TUTORIAL - SPRINKLER

% Estructura de la red bayesiana
N = 4; C = 1; S = 2; R = 3; W = 4;
grafo = zeros(N, N);
grafo(C, [R S]) = 1;
grafo([R S], W) = 1;

% La talla de los nodos es el cardinal del conjunto de valores que puede
% tomar cada nodo, en este caso son todos discretos y binarios
nodosDiscretos = 1:N;
tallaNodos = 2*ones(1, N);

% Make net
redB = mk_bnet(grafo, tallaNodos, 'discrete', nodosDiscretos)

% Constructor de TPC
redB.CPD{W} = tabular_CPD(redB, W, [1.0 0.1 0.1 0.01 0.0 0.9 0.9 0.99]);
redB.CPD{C} = tabular_CPD(redB, C, [0.5 0.5]);
redB.CPD{S} = tabular_CPD(redB, S, [0.5 0.9 0.5 0.1]);
redB.CPD{R} = tabular_CPD(redB, R, [0.8 0.2 0.2 0.8]);

% Motor de inferencia
motor = jtree_inf_engine(redB);

% Evidencia
evidencia = cell(1, N); % vector multidimensional
evidencia{W} = 2; % esta mojado
evidencia{R} = 2; % esta lloviendo
%evidencia{S} = 2; % prueba absurda

% Insercion de la evidencia
[motor, logVerosim] = enter_evidence(motor, evidencia);

% Calculo de P(S=s|W=2) para s€{false, true}
m = marginal_nodes(motor, S, 1);
% el 1 del final es opcional, vale para para la prueba absurda
m.T

% Distribucion conjunta: P(S=s,R=r,W=w|evidencia)
evidencia = cell(1, N);
evidencia{R} = 2;
[motor, logVerosim] = enter_evidence(motor, evidencia);
m = marginal_nodes(motor, [S R W], 1);
% de nuevo el 1 hace que se printeen los ceros
m.T

% Explicacion mas probable
evidencia = cell(1, N);
[explMaxProb, logVerosim] = calc_mpe(motor, evidencia)
probabilidad = 2^logVerosim

%% Aprendizaje

% Generacion aleatoria
semilla = 0; rng(semilla);
nMuestras = 100;
muestras = cell(N, nMuestras);
for i=1:nMuestras
    muestras(:,i) = sample_bnet(redB);
end

% Nueva red
redAPR = mk_bnet(grafo, tallaNodos);
redAPR.CPD{W} = tabular_CPD(redAPR, W);
redAPR.CPD{C} = tabular_CPD(redAPR, C);
redAPR.CPD{S} = tabular_CPD(redAPR, S);
redAPR.CPD{R} = tabular_CPD(redAPR, R);

redAPR2 = learn_params(redAPR, muestras);

% Probabilidades estimadas
TPCaux = cell(1,N);
for i=1:N
    s = struct(redAPR2.CPD{i});
    TPCaux{i} = s.CPT;
end

dispcpt(TPCaux{W})

%% Aprendizaje con datos incompletos mediante EM

% Ocultamos el 50% de los datos aleatoriamente
muestrasS = muestras;
semilla = 0; rng(semilla);
ocultas = rand(N, nMuestras) > 0.5;
[I, J] = find(ocultas);
for k=1:length(I)
    muestrasS{I(k), J(k)} = [];
end

% Nueva red
redEM = mk_bnet(grafo, tallaNodos);
redEM.CPD{W} = tabular_CPD(redEM, W);
redEM.CPD{C} = tabular_CPD(redEM, C);
redEM.CPD{S} = tabular_CPD(redEM, S);
redEM.CPD{R} = tabular_CPD(redEM, R);
motorEM = jtree_inf_engine(redEM);

% Aprendizaje EM
maxIter = 100; eps = 1e-4;
semilla = 0; rng(semilla);
[redEM2, trazaLogVer] = learn_params_em(motorEM, muestrasS, maxIter, eps);
auxTPC = cell(1, N);
for i=1:N
    s = struct(redEM2.CPD{i});
    auxTPC{i} = s.CPT;
end

dispcpt(auxTPC{S})
dispcpt(auxTPC{W})
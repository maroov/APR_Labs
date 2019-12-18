%% APR - Práctica 4: Redes Bayesianas
%% Nahuel Unai Roselló Beneitez
%% Manuel Roselló Oviedo

%% EJERCICIO 1 - SPRINKLER

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
redB = mk_bnet(grafo, tallaNodos, 'discrete', nodosDiscretos);

% Constructor de TPC
redB.CPD{W} = tabular_CPD(redB, W, [1.0 0.1 0.1 0.01 0.0 0.9 0.9 0.99]);
redB.CPD{C} = tabular_CPD(redB, C, [0.5 0.5]);
redB.CPD{S} = tabular_CPD(redB, S, [0.5 0.9 0.5 0.1]);
redB.CPD{R} = tabular_CPD(redB, R, [0.8 0.2 0.2 0.8]);

%% Aprendizaje

% Generacion aleatoria
semilla = 0; rng(semilla);
nMuestras = 100;
muestras = cell(N, nMuestras);
for i=1:nMuestras
    muestras(:,i) = sample_bnet(redB);
end

%% DATOS COMPLETOS

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

disp("Datos completos");
disp("W:");
dispcpt(TPCaux{W})
disp("S:");
dispcpt(TPCaux{S})
disp("R:");
dispcpt(TPCaux{R})
disp("C:");
dispcpt(TPCaux{C})

%% DATOS INCOMPLETOS

% Ocultamos el 50% de los datos aleatoriamente (nueva semilla)
muestrasS = muestras;
semilla = 3; rng(semilla);
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
maxIter = 1000; eps = 1e-4;
semilla = 0; rng(semilla);
[redEM2, trazaLogVer] = learn_params_em(motorEM, muestrasS, maxIter, eps);
TPCaux = cell(1, N);
for i=1:N
    s = struct(redEM2.CPD{i});
    TPCaux{i} = s.CPT;
end

disp("Datos incompletos");
disp("W:");
dispcpt(TPCaux{W})
disp("S:");
dispcpt(TPCaux{S})
disp("R:");
dispcpt(TPCaux{R})
disp("C:");
dispcpt(TPCaux{C})

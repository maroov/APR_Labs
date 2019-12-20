%% APR - Práctica 4: Redes Bayesianas
%% Nahuel Unai Roselló Beneitez
%% Manuel Roselló Oviedo

%% EJERCICIO 2 - CANCER DE PULMON

%% B.1
% Estructura de la red bayesiana
N = 5; P = 1; F = 2; C = 3; R = 4; D = 5;
grafo = zeros(N, N);
grafo([P F], C) = 1;
grafo(C, [R D]) = 1;

% La talla de los nodos es el cardinal del conjunto de valores que puede
% tomar cada nodo, en este caso seran:
nodosDiscretos = 1:N;
tallaNodos = [2 2 2 3 2];

% Make net
redB = mk_bnet(grafo, tallaNodos, 'discrete', nodosDiscretos);

% Constructor de TPC
% Polucion: b (bajo), a (alto)
redB.CPD{P} = tabular_CPD(redB, P, [0.9 0.1]);
% Fumador: n (no), s (si)
redB.CPD{F} = tabular_CPD(redB, F, [0.7 0.3]);
% Cancer: n (negativo), p (positivo)
redB.CPD{C} = tabular_CPD(redB, C, [0.999 0.97 0.95 0.92 0.001 0.03 0.05 0.08]);
% Rayos X: n (negativo), d (dudoso), p (positivo)
redB.CPD{R} = tabular_CPD(redB, R, [0.8 0.1 0.1 0.2 0.1 0.7]);
% Disnea n (no), s (si)
redB.CPD{D} = tabular_CPD(redB, D, [0.7 0.35 0.3 0.65]);


%% B.2
disp("Ejercicio B.2");
% P(!C | R = n, D = s)
evidencia = cell(1, N);
evidencia{R} = 1; % Rayos no
evidencia{D} = 2; % Disnea si
motor = jtree_inf_engine(redB);
[motor, logVerosim] = enter_evidence(motor, evidencia);
m = marginal_nodes(motor, C, 1);
disp(" ");
disp("P(!C | R = n, D = s)");
m.T(1)

%% B.3
disp("Ejercicio B.3");
% Explicacion mas probable de cancer
evidencia = cell(1, N);
evidencia{C} = 2;
[explMaxProb, logVerosim] = calc_mpe(motor, evidencia);
disp("Explicacion mas probable de cancer:");
explMaxProb
logVerosim
probabilidad = exp(logVerosim)
% 1 2 2 3 2
% Polucion baja, Fumador si, Cancer positivo, Rayos X positivo, Disnea si


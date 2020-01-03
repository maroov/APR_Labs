%% APR - Tema 6: Modelos gráficos
%% Manuel Roselló Oviedo

%% ENTREGABLE 6 - CANCER DE PULMON

% Estructura de la red bayesiana
N = 5; P = 1; F = 2; C = 3; X = 4; D = 5;
grafo = zeros(N, N);
grafo([P F], C) = 1;
grafo(C, [X D]) = 1;

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
redB.CPD{X} = tabular_CPD(redB, X, [0.8 0.1 0.1 0.1 0.2 0.7]);
% Disnea n (no), s (si)
redB.CPD{D} = tabular_CPD(redB, D, [0.7 0.35 0.3 0.65]);


%% 1)   la probabilidad de que el paciente sea fumador 
%%      sabiendo que padece disnea y que los resultados 
%%      de rayos X han salido negativos
disp("Ejercicio 1");
% P(F = s | X = n, D = s)
evidencia = cell(1, N);
evidencia{X} = 1; % Rayos negativo
evidencia{D} = 2; % Disnea si
motor = jtree_inf_engine(redB);
[motor, logVerosim] = enter_evidence(motor, evidencia);
m = marginal_nodes(motor, F, 1);
disp(" ");
disp("P(F = s | X = n, D = s)");
m.T(2)

%% 2)   la probabilidad de que un paciente sufra disnea 
%%      sabiendo que es fumador y que los resultados 
%%      de rayos X han salido positivos
disp("Ejercicio 2");
% P(D = s | X = p, F = s)
evidencia = cell(1, N);
evidencia{X} = 3; % Rayos positivo
evidencia{F} = 2; % Fumador si
motor = jtree_inf_engine(redB);
[motor, logVerosim] = enter_evidence(motor, evidencia);
m = marginal_nodes(motor, D, 1);
disp(" ");
disp("P(D = s | X = p, F = s)");
m.T(2)

%% 3)   a probabilidad de que un paciente sufra cáncer 
%%      sabiendo que es fumador, sufre disnea y que los
%%      resultados de rayos X han salido positivos
disp("Ejercicio 3");
% P(C = s | D = s, X = s, F = s)
evidencia = cell(1, N);
evidencia{X} = 3; % Rayos positivo
evidencia{F} = 2; % Fumador si
evidencia{D} = 2; % Disnea si
motor = jtree_inf_engine(redB);
[motor, logVerosim] = enter_evidence(motor, evidencia);
m = marginal_nodes(motor, C, 1);
disp(" ");
disp("P(C = s | D = s, X = s, F = s)");
m.T(2)

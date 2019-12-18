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
redAPR.CPD{P} = tabular_CPD(redAPR, P);
redAPR.CPD{F} = tabular_CPD(redAPR, F);
redAPR.CPD{C} = tabular_CPD(redAPR, C);
redAPR.CPD{R} = tabular_CPD(redAPR, R);
redAPR.CPD{D} = tabular_CPD(redAPR, D);

redAPR2 = learn_params(redAPR, muestras);

% Probabilidades estimadas
TPCaux = cell(1,N);
for i=1:N
    s = struct(redAPR2.CPD{i});
    TPCaux{i} = s.CPT;
end

disp("Datos completos");
disp("P:");
dispcpt(TPCaux{P})
disp("F:");
dispcpt(TPCaux{F})
disp("C:");
dispcpt(TPCaux{C})
disp("R:");
dispcpt(TPCaux{R})
disp("D:");
dispcpt(TPCaux{D})

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
redEM.CPD{P} = tabular_CPD(redEM, P);
redEM.CPD{F} = tabular_CPD(redEM, F);
redEM.CPD{C} = tabular_CPD(redEM, C);
redEM.CPD{R} = tabular_CPD(redEM, R);
redEM.CPD{D} = tabular_CPD(redEM, D);
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
disp("P:");
dispcpt(TPCaux{P})
disp("F:");
dispcpt(TPCaux{F})
disp("C:");
dispcpt(TPCaux{C})
disp("R:");
dispcpt(TPCaux{R})
disp("D:");
dispcpt(TPCaux{D})
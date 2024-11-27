clear all;
%unzip("rice.zip");

data = imageDatastore("rice", IncludeSubfolders = true, LabelSource = "foldernames");

classNames = categories(data.Labels);
%labelCount = countEachLabel(data);
 
[dataTrain, dataValidation, dataTest, dataPlay] = splitEachLabel(data, 0.7, 0.14, 0.15, 0.01, "randomized");
%splitEachLabel(data, 0.7, 0.14, 0.15, 0.01, "randomized");


dataTrain.ReadFcn=@(filename) im2gray(imread(filename));
dataValidation.ReadFcn=@(filename) im2gray(imread(filename));
dataTest.ReadFcn=@(filename) im2gray(imread(filename));
dataPlay.ReadFcn=@(filename) im2gray(imread(filename));

% Definicja warstw sieci
layers = [
    imageInputLayer([250 250 1])   % Warstwa wejściowa
    
    % convolution2dLayer(3, 32, "Padding", 0)
    % reluLayer
    % maxPooling2dLayer(5, 'Stride', 3)
    % 

    flattenLayer   % Spłaszczenie do wektora
    fullyConnectedLayer(512)
    reluLayer

    fullyConnectedLayer(256)
    reluLayer

    fullyConnectedLayer(5)   % Warstwa wyjściowa dla 5 klas
    softmaxLayer             % Softmax do klasyfikacji wieloklasowej
    classificationLayer];    % Warstwa klasyfikacji

% Analiza sieci (opcjonalnie)
analyzeNetwork(layers);

% Konfiguracja treningu
options = trainingOptions("sgdm", ...
    InitialLearnRate=0.00001, ...
    MaxEpochs=6, ...
    Shuffle="once", ...
    ValidationData=dataValidation, ...
    ValidationFrequency=30, ...
    Plots="training-progress", ... ...
    Verbose=false, ...
    ExecutionEnvironment="auto");


% Trening sieci
net = trainNetwork(dataTrain,layers,options);
%net = trainnet(dataTrain, layers, options);

% % Klasyfikacja na zbiorze walidacyjnym
YPred = classify(net, dataValidation);
YValidation = dataValidation.Labels;
% 
% % Obliczanie dokładności
accuracy = sum(YPred == YValidation) / numel(YValidation);
% 
% % Klasyfikacja na zbiorze testowym
YTest = classify(net, dataTest);

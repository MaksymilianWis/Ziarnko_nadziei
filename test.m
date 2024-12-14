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
    
    convolution2dLayer(3, 32, "Padding", 0)
    reluLayer
    maxPooling2dLayer(5, 'Stride', 3)
    
    flattenLayer   % Spłaszczenie do wektor
    
    fullyConnectedLayer(1024)
    reluLayer

    fullyConnectedLayer(512)
    reluLayer

    fullyConnectedLayer(256)
    reluLayer

    fullyConnectedLayer(5)   % Warstwa wyjściowa dla 5 klas
    softmaxLayer             % Softmax do klasyfikacji wieloklasowej
    classificationLayer];    % Warstwa klasyfikacji

analyzeNetwork(layers);

options = trainingOptions("adam", ...
    InitialLearnRate=0.00001, ...
    MaxEpochs=5, ...
    Shuffle="every-epoch", ...
    ValidationData=dataValidation, ...
    ValidationFrequency=30, ...
    Verbose=false, ...
    ExecutionEnvironment="auto", ...
    Plots="training-progress");


%trenowanie sieci
net = trainNetwork(dataTrain,layers,options);


%======= dokladnosc na zbiorze walidacyjnym
YPred = classify(net, dataValidation);
YValidation = dataValidation.Labels;
accuracy = sum(YPred == YValidation) / numel(YValidation);

%======= dokladnosc na test zbiorze
YPred_Test = classify(net, dataTest);
YTest=dataTest.Labels;
accuracyTest = sum(YPred_Test==YTest) / numel(YTest);
disp("TESTaccuracy:  "+ accuracyTest);

%========= zapis
%save("trainedNetwork.mat", "net");
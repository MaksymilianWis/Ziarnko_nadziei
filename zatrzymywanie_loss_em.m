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
    
    flattenLayer   % Spłaszczenie do wektora

    fullyConnectedLayer(512)
    reluLayer

    fullyConnectedLayer(256)
    reluLayer

    fullyConnectedLayer(5)   % Warstwa wyjściowa dla 5 klas
    softmaxLayer             % Softmax do klasyfikacji wieloklasowej
    classificationLayer];    % Warstwa klasyfikacji

analyzeNetwork(layers);


crossEntropyThreshold = 0.1;
%ValidationPatience jezeli validation nie spada przez x epok zatrzymywanie
%==============
options = trainingOptions("sgdm", ...
    InitialLearnRate=0.00001, ...
    MaxEpochs=10, ...
    Shuffle="every-epoch", ...
    ValidationData=dataValidation, ...
    ValidationFrequency=30, ...
    Verbose=false, ...
    ExecutionEnvironment="auto", ...
    Plots="training-progress", ...
    OutputFcn=@(info) stopTrainingOnCrossEntropy(info, crossEntropyThreshold));
%==============

%trenowanie sieci
net= trainNetwork(dataTrain,layers,options);

%======= dokladnosc na zbiorze walidacyjnym
YPred = classify(net, dataValidation);
YValidation = dataValidation.Labels;
accuracy = sum(YPred == YValidation) / numel(YValidation);

%======= dokladnosc na test zbiorze
YPred_Test = classify(net, dataTest);
YTest=dataTest.Labels;
accuracyTest = sum(YPred_Test==YTest) / numel(YTest);
disp("TESTaccuracy:  "+ accuracyTest);

figure;
confusionchart(YTest, YPred_Test);
title('Macierz Trafień - Zbiór Testowy');


%=======
% Funkcja zatrzymania na podstawie cross-entropy loss
function stop = stopTrainingOnCrossEntropy(info, threshold)
    stop = false;

    % Sprawdzenie wartości straty walidacyjnej
    if ~isempty(info.ValidationLoss)
        currentLoss = info.ValidationLoss;
        disp("Validation Cross-Entropy Loss: " + currentLoss);

        % Zatrzymanie, jeśli cross-entropy loss spadnie poniżej progu
        if currentLoss < threshold
            disp("Zatrzymanie: osiągnięto próg cross-entropy loss " + threshold);
            stop = true;
        end
    end
end


%========= zapis
%save("trainedNetwork.mat", "net");
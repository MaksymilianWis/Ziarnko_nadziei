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
lossThreshold = 0.01;
options = trainingOptions("sgdm", ...
    InitialLearnRate=0.00001, ...
    MaxEpochs=10, ...
    Shuffle="every-epoch", ...
    ValidationData=dataValidation, ...
    ValidationFrequency=30, ...
    Verbose=false, ...
    ExecutionEnvironment="auto", ...
    Plots="training-progress", ...
    OutputFcn=@(info)stopTraining(info,lossThreshold) ...
    );


%trenowanie sieci
[net, info] = trainNetwork(dataTrain,layers,options);

trainingLoss = info.TrainingLoss;
validationLoss = info.ValidationLoss;

% Create a custom plot
figure;
plot(trainingLoss, 'LineWidth', 1.5);
hold on;
plot(validationLoss, 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Loss');
title('Training and Validation Loss');
legend('Training Loss', 'Validation Loss');
ylim([0, max(trainingLoss) * 1.2]); % Adjust y-axis scaling
grid on;

%======= dokladnosc na zbiorze walidacyjnym
YPred = classify(net, dataValidation);
YValidation = dataValidation.Labels;
accuracy = sum(YPred == YValidation) / numel(YValidation);

%======= dokladnosc na test zbiorze
YPred_Test = classify(net, dataTest);
YTest=dataTest.Labels;
accuracyTest = sum(YPred_Test==YTest) / numel(YTest);
disp("TESTaccuracy:  "+ accuracyTest);

function stop = stopTraining(info, lossThreshold)
    persistent counterBelowThreshold
    if isempty(counterBelowThreshold)
        counterBelowThreshold = 0; % licznik inicjalizowany raz
    end

    trainingLoss = info.TrainingLoss;
    if ~isempty(trainingLoss) && trainingLoss < lossThreshold
        counterBelowThreshold = counterBelowThreshold + 1;
    else
        counterBelowThreshold = 0; % reset, jeśli loss nie spełnia kryterium
    end

    % Zatrzymaj tylko wtedy, gdy loss spełnia kryterium przez 3 kolejne iteracje
    stop = counterBelowThreshold >= 3;
end
% Extract loss data


%========= zapis
%save("trainedNetwork.mat", "net");
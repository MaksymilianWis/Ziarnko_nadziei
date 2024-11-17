clear all;
%unzip("rice.zip");

data = imageDatastore("rice", IncludeSubfolders = true, LabelSource = "foldernames");

classNames = categories(data.Labels);

[dataTrain, dataValidation, dataTest, dataPlay] = splitEachLabel(data, 0.7, 0.14, 0.15, 0.01);

layers = [
    imageInputLayer([250 250 3])

    convolution2dLayer(5, 64, 'Padding','same')

    fullyConnectedLayer(5)
    softmaxLayer
    classificationLayer];

analyzeNetwork(layers);

options = trainingOptions('sgdm','InitialLearnRate', 0.05, ...
                            'MaxEpochs', 1, ...
                            'shuffle', 'once', ...
                            'ValidationData', dataValidation, ...
                            'ValidationFrequency', 10, ...
                            'Verbose', false, ...
                            'Plots','training-progress');

net = trainNetwork(dataTrain, layers, options);

YPred = classify(net, dataValidation);
YValidation = dataValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);

YTest = predict(net, dataTest);
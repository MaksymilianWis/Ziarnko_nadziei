clear all;
%unzip("rice.zip");

data = imageDatastore("rice", IncludeSubfolders = true, LabelSource = "foldernames");

classNames = categories(data.Labels);

[dataTrain, DataValidation, dataTest, dataPlay] = splitEachLabel(data, 0.7, 0.14, 0.15, 0.01);

layers = [
    imageInputLayer([250 250 1])

    convolution2Layer(5, 64, 'Padding','same')
]
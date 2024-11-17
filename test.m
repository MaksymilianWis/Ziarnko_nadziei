%unzip("rice.zip");

data = imageDatastore("rice", IncludeSubfolders = true, LabelSource = "foldernames");

classNames = categories(data.Labels);

[dataTrain, DataValidation, dataTest, dataPlay] = splitEachLabel(data, 0.7, 0.14, 0.15, 0.01);


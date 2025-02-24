% This code is implemented for the paper: CardioECGNet: A novel deep learning architecture for accurate and automated ECG signal 
% classification across diverse cardiac conditions. Published in Biomedical Signal Processing and Control
% Volume 106, August 2025, 107720
%% Authors: Mohamed Hammad, Mohammed ElAffendi, Ahmed A. Abd El-Latif
%

clc
clear
close all

% Load the layer graph from TestCallfun.m
run('TestCallCastumFun.m');

% Check connectivity of the layers in the layer graph
connections = lgraph.Connections;

% Display the connections to identify any issues
 disp(connections);

% Connect the output layer to the preceding layer if necessary
if ~any(strcmp(connections.Destination, 'Output'))
    lgraph = connectLayers(lgraph, 'PurkinjeFibers', 'Output');
end

% Define parameters for training
dataECG = fullfile('F:\ECG Data\COVID-19');
imds2 = imageDatastore(dataECG, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

[testSet,trainingSet] = splitEachLabel(imds2, 0.2, 'randomize');
inputSize=[200 200];

%Data augmentation for the training set
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandRotation', [-10 10], ...
    'RandXScale', [0.7 1.3], ...
    'RandYScale', [0.7 1.3], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10], ...
    'RandXShear', [-20 20], ...
    'RandYShear', [-20 20]);

%Create augmented image datastore
trainingSet1 = augmentedImageDatastore(inputSize, trainingSet, ...
    'ColorPreprocessing', 'rgb2gray', ...
    'DataAugmentation', augmenter);

% Create augmented image datastore for the test set
testSet1 = augmentedImageDatastore(inputSize, testSet, ...
    'ColorPreprocessing', 'rgb2gray');


% Define training options
train_options = trainingOptions('adam', ...
    'MiniBatchSize', 8, ...
    'MaxEpochs', 30, ...
    'L2Regularization', 0.0005, ...
    'InitialLearnRate', 3.0000000e-04, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 5, ...
    'ValidationData', testSet1, ...
    'Shuffle', 'every-epoch',...
    'ValidationFrequency', 87, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

% Start timer
tic;

% Train the model
[net1, info] = trainNetwork(trainingSet1, lgraph, train_options);

% Plot training progress with a red line for early stopping
figure;
plot(info.TrainingLoss, '-b', 'LineWidth', 1.5);
hold on;
plot(info.ValidationAccuracy, '-g', 'LineWidth', 1.5);
if isfield(info, 'Epoch') && isfield(info, 'ValidationAccuracy')
    idx = find(info.ValidationAccuracy >= 0.90, 1);
    if ~isempty(idx)
        xline(idx, '--r', 'LineWidth', 2, 'Label', 'Early Stop');
    end
end
xlabel('Epochs');
ylabel('Loss / Accuracy');
legend('Training Loss', 'Validation Accuracy', 'Location', 'best');
title('Training Progress');
grid on;
hold off;
%%

YPred = classify(net1,testSet1);

% Stop timer
elapsed_time = toc;
fprintf('Elapsed time for processing: %.2f seconds\n', elapsed_time);


% Count the number of parameters in the network
% Initialize total number of parameters
num_params = 0;

% Loop through each layer in the network
layers = net1.Layers;
for i = 1:numel(layers)
    if isprop(layers(i), 'Weights') && ~isempty(layers(i).Weights)
        % Add the number of parameters in the weights and biases of the layer
        num_params = num_params + numel(layers(i).Weights) + numel(layers(i).Bias);
    end
end

fprintf('Number of parameters in the network: %d\n', num_params);

num_params = sum(arrayfun(@(x) numel(x.Weights) + numel(x.Bias), ...
    net1.Layers(arrayfun(@(x) isprop(x, 'Weights') && ~isempty(x.Weights), net1.Layers))));
fprintf('Number of parameters in the network: %d\n', num_params);

% Evaluate the model
accuracy = sum(YPred == testSet.Labels) / numel(testSet.Labels);
% fprintf('Accuracy: %.2f%%\n', accuracy * 100);

%%
% Calculate the confusion matrix and metrics
confMat = confusionmat(testSet.Labels, YPred);
[precision, recall, f1Score, specificity, sensitivity] = deal(zeros(1,2));

classLabels = {'covid-19','normal'};

% Create a confusion matrix chart
confMatrixChart = confusionchart(confMat, classLabels, ...
    'Title', 'Confusion Matrix', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

% Customize the confusion matrix chart appearance
confMatrixChart.FontSize = 12;
confMatrixChart.Normalization = 'absolute';
confMatrixChart.RowSummary = 'row-normalized';
confMatrixChart.ColumnSummary = 'column-normalized';

for i = 1:2
    TP = confMat(i,i);
    FP = sum(confMat(:,i)) - TP;
    FN = sum(confMat(i,:)) - TP;
    TN = sum(confMat(:)) - TP - FP - FN;
    precision(i) = TP / (TP + FP);
    recall(i) = TP / (TP + FN);
    f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
    specificity(i) = TN / (TN + FP);
    sensitivity(i) = recall(i);
end

disp('Precision: '); disp(precision);
disp('Recall: '); disp(recall);
disp('F1 Score: '); disp(f1Score);
disp('Specificity: '); disp(specificity);
disp('Sensitivity: '); disp(sensitivity);

figure;
bar([accuracy, mean(precision), mean(recall), mean(sensitivity), mean(specificity), mean(f1Score)]);
set(gca, 'XTickLabel', {'Accuracy', 'Precision', 'Recall', 'Sensitivity', 'Predictivity', 'F1Score'});
title('Performance Metrics');
ylabel('Value');

%%

% Visualize activations of each layer
img = testSet1.Files{1};
img = imread(img);
img = rgb2gray(img);

layersToVisualize = {'SANodeInput', 'AVNode', 'BundleOfHis', 'PurkinjeFibers'};
for layerName = layersToVisualize
    act = activations(net1, img, layerName{1});
    act = mat2gray(act);
    act = imtile(act, 'ThumbnailSize', [64 64]);
    figure;
    imshow(act);
    title(['Layer ', layerName{1}, ' Features'], 'Interpreter', 'none');
end

layers = lgraph.Layers;
num_params = 0;

fprintf('Layer-wise parameter count:\n');
for i = 1:numel(layers)
    layer = layers(i);
    layer_params = 0;
    
    if isprop(layer, 'Weights') && ~isempty(layer.Weights)
        layer_params = layer_params + numel(layer.Weights);
    end
    if isprop(layer, 'Bias') && ~isempty(layer.Bias)
        layer_params = layer_params + numel(layer.Bias);
    end
    
    fprintf('%s: %d parameters\n', layer.Name, layer_params);
    num_params = num_params + layer_params;
end

fprintf('Total number of parameters in the network: %d\n', num_params);

% After computing metrics (precision, recall, etc.)
%% Statistical Analysis - T-Test and Bootstrap
% Perform T-Test and Visualization
class1 = find(testSet.Labels == 'covid-19');
class2 = find(testSet.Labels == 'normal');

YPredProb = predict(net1, testSet1); % Predicted probabilities
class1_probs = YPredProb(class1, 1); % Probabilities for Arrhythmia
class2_probs = YPredProb(class2, 2); % Probabilities for Normal

[h_ttest, p_ttest] = ttest2(class1_probs, class2_probs, 'Vartype', 'unequal');
fprintf('Two-Sample T-Test:\nHypothesis Rejected (1 = Yes, 0 = No): %d\nP-Value: %.4f\n', h_ttest, p_ttest);

% Bootstrap Analysis for Accuracy
n_boot = 1000; % Number of bootstrap samples
bootstrap_acc = zeros(1, n_boot);

for i = 1:n_boot
    idx = randi(numel(testSet.Labels), numel(testSet.Labels), 1);
    YPred_boot = YPred(idx);
    Labels_boot = testSet.Labels(idx);
    bootstrap_acc(i) = sum(YPred_boot == Labels_boot) / numel(Labels_boot);
end

conf_int = prctile(bootstrap_acc, [2.5, 97.5]);
fprintf('Bootstrap Accuracy Analysis:\nMean Accuracy: %.4f\n95%% Confidence Interval: [%.4f, %.4f]\n', mean(bootstrap_acc), conf_int(1), conf_int(2));

% Display T-Test Results
tTestTable = table({'covid-19'; 'normal'}, [mean(class1_probs); mean(class2_probs)], ...
    [std(class1_probs); std(class2_probs)], 'VariableNames', {'Class', 'Mean Probability', 'Std Deviation'});
disp(tTestTable);

% Plot Histogram for Bootstrap Results
figure;
histogram(bootstrap_acc, 30, 'Normalization', 'pdf');
hold on;
xline(mean(bootstrap_acc), 'r', 'LineWidth', 2, 'Label', 'Mean Accuracy');
xline(conf_int(1), '--k', 'LineWidth', 2, 'Label', '2.5% CI');
xline(conf_int(2), '--k', 'LineWidth', 2, 'Label', '97.5% CI');
legend('Bootstrap Accuracy', 'Mean', 'Confidence Interval');
title('Bootstrap Accuracy Distribution');
xlabel('Accuracy');
ylabel('Density');
grid on;
hold off;






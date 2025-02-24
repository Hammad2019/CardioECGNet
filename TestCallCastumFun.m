%% This code for the Paper: CardioECGNet: A novel deep learning architecture for accurate and automated ECG signal classification across diverse cardiac conditions
%% Edited by Mohamed Hammad
%%
% Define parameters
inputSize = [200 200]; % Example input size
numChannels = 3; % Example number of input channels
numClasses = 2; % Example number of output classes


% Define parameters for the custom layers
V_rest = -70;  % Resting membrane potential in millivolts (mV)
V_peak = 30;   % Peak membrane potential in millivolts (mV)
D_max = 20;    % Maximum delay in milliseconds (ms)
t_0 = 50;       % Time at which delay is introduced in milliseconds (ms)
tau = 1;       % Time constant in milliseconds (ms)
g = 5;       % Membrane conductance

% Create the custom layers
InputLayer = imageInputLayer(inputSize,'Name','Input');

sanodeLayer = SANodeLayer('SANodeInput', V_rest, V_peak); % You need to define V_rest and V_peak
avnodeLayer = AVNodeLayer('AVNode', D_max, t_0, tau); % You need to define D_max, t_0, and tau
bundleOfHisLayer = BundleOfHisLayer('BundleOfHis', g); % You need to define g
purkinjeFibersLayer = PurkinjeFibersLayer('PurkinjeFibers', g); % You need to define g

Layers2 =[
 fullyConnectedLayer(numClasses,'Name','Full');

% Define the softmax and classification layers
softmaxLayer('Name','Soft');
classificationLayer('Name','Output')];

% Create the layer graph
lgraph = layerGraph();

% Add the layers to the layer graph
lgraph = addLayers(lgraph, InputLayer);
lgraph = addLayers(lgraph, sanodeLayer);
lgraph = addLayers(lgraph, avnodeLayer);
lgraph = addLayers(lgraph, bundleOfHisLayer);
lgraph = addLayers(lgraph, purkinjeFibersLayer);
lgraph = addLayers(lgraph, Layers2);


% Connect the layers
lgraph = connectLayers(lgraph, 'Input', 'SANodeInput');
lgraph = connectLayers(lgraph, 'SANodeInput', 'AVNode');
lgraph = connectLayers(lgraph, 'AVNode', 'BundleOfHis');
lgraph = connectLayers(lgraph, 'BundleOfHis', 'PurkinjeFibers');
lgraph = connectLayers(lgraph, 'PurkinjeFibers', 'Full');


% View the network
plot(lgraph)

% Count parameters


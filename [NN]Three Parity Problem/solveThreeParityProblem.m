% Execution Time (Total Learning time for all configurations) is approximately 6 minutes
% Program was written and tested using GNU Octave

% Each training pattern represents a column in x matrix
x=[0 0 0 0 1 1 1 1
   0 0 1 1 0 0 1 1
   0 1 0 1 0 1 0 1
  ];

t=[0 1 1 0 1 0 0 1];  

tic;

% Use Sigmoid as the activation function of the output neuron
% Try different no of neurons in the hidden layer

for hiddenNeurons=1:10
  error=backPropagation(t,x,hiddenNeurons,75000,1)
end
display('Optimal No of neurons in the hidden layer was heuristically found to be 4')

% Use linear activation function in the output neuron
LinearNeuronError=backPropagation(t,x,4,75000,0)

toc;

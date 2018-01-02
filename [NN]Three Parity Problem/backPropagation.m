function [error]=backPropagation(t,x,hiddenNeurons,iterations,outputIsSigmoid)
  % This function implements the stochastic gradient descent learning algorithm 
  % t -> Targets for the 8 training patterns  [1*8]
  % x -> The 8 training patterns  [3*8]
  % hiddenNeurons -> No. of neurons in the hidden layer 
  % iterations -> No. of Epochs
  % outputIsSigmoid -> Choose the activation function of the output neuron (Sigmoid/Linear)
  
  % All the weights are initialized randomly in range [-0.2,0.2]
  W1=(0.4*rand(hiddenNeurons,3))-0.2*ones(hiddenNeurons,3);
  W2=(0.4*rand(1,hiddenNeurons))-0.2*ones(1,hiddenNeurons);
  B1=(0.4*rand(hiddenNeurons,1))-0.2*ones(hiddenNeurons,1);
  B2=(0.4*rand(1,1))-0.2;
  % Relatively small Learning rate
  eta=0.3;  
 
  for it=1:iterations
    for pattern=1:8
      % Forward Path
      I1=x(:,pattern);
      net1=(W1*I1)+B1;
      O1=1./(1+exp(-net1));
      I2=O1;
      net2=W2*I2+B2;
      if(outputIsSigmoid)
        O2=1./(1+exp(-net2));
      else
        O2=net2;
      end
      % Backward Path + Weight Update
      if(outputIsSigmoid)
        delta2=-(t(pattern)-O2).*(O2.*(1-O2));
      else 
        delta2=-(t(pattern)-O2);
      end
      delta1=(O1.*(1-O1)).*((W2')*delta2);   
      W2=W2-(eta*(delta2*O1'));
      B2=B2-(eta.*delta2);
      W1=W1-(eta*delta1*I1');
      B1=B1-(eta*delta1);
    end
  end
 
  % Find the total final Error
  error=0;
  for pattern=1:8
    I1=x(:,pattern);
    net1=W1*I1+B1;
    O1=1./(1+exp(-net1));
    I2=O1;
    net2=W2*I2+B2;
    if(outputIsSigmoid)
      O2=1./(1+exp(-net2));
    else
      O2=net2;
    end
    error=error+(O2-t(pattern))*(O2-t(pattern));
  end
  error=0.5*error;

end

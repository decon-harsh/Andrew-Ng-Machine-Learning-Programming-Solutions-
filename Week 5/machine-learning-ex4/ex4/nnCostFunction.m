function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%


X=[ones(m,1) , X];

a1=X;     % 5000 X 401
z2=a1*Theta1'; % 5000 x 25
a2=sigmoid(z2); % 5000 x 25
a2=[ones(size(a2,1),1) , a2]; %5000 X 26  
z3=a2*Theta2'; %5000 X 10 
a3=sigmoid(z3);
h=a3; 
%Forward propagartion ends

%Vectorising y
y_vector= (1:num_labels)==y;
%its like making y a matrix of 5000(number of entries) x 10(output classes)
%why am i doing this?
%because this is a vectorise implementation of the seconf summation thing in cost function 
%summation 1 to k can be reduced by element wise wise operations on this y_vector

%Cost function without regularization
J=  (1/m)*sum(sum(-y_vector.*log(h)-(1-y_vector).*log(1-h)));
%adding regularization;
J=J+((lambda)/(2*m))*((sum(sum(Theta1(:,2:end).^2)))+(sum(sum(Theta2(:,2:end).^2))));


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

delt3=a3-y_vector; %5000 x 10
delt2=((delt3)*Theta2).*[ones(size(z2,1),1) sigmoidGradient(z2)]; %5000x26 %adding ones for the compensation of bias unit in activations
delt2=delt2(:,2:end); %removing bias term %5000 x 25

Theta1_grad=Theta1_grad+(1/m)*(delt2'*a1);
Theta2_grad=Theta2_grad+(1/m)*(delt3'*a2);



% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+(lambda/m)*(Theta1(:,2:end));
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+(lambda/m)*(Theta2(:,2:end));



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


%                                       Starts here                                    %

%By looking at the ex3-pdf file we got to know 
%Dimensions
%X is m X n
%theta is n X 1
% y must me m X 1 as X has n features each having m training examples so each training example will have one output

%Step1 calculate sigmoid function
h=sigmoid(X*theta); %done 

%Step2 simply claculate J
J=(1/m)*(sum((-y.*log(h))-((1-y).*log(1-h)))) %used each element operator  
%(-y.*log(h)) will look something like this [-y1*log(h1);-y2*loh(h2);.....]

%Step3 calculate gradients
grad=(1/m)*((X)'*(h-y))
%In this step we saw the dimensions first 
%We wanted to make something like [(h1-Y1)*X1+(h2-y2)*X2] all X is one of one feature like example length of piece of plot to be sold only length not breadth , in next repeat same for breadth (other features)


%Step4 make additional terms
additional_term1=(lambda/(2*m))*sum(theta(2:end).^2); %why 2:end because thetha0 is bias term and it has to be left 
additional_term2=(lambda/m)*(theta(2:end));



%Final touchup
%add additional terms
J=J+additional_term1;
grad(2:end)=grad(2:end)+additional_term2;%because grad(0) is left witout adding anything
 

%ready to submit?







% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%










% =============================================================

grad = grad(:);% this is adding grad(1) and other grad(2:end) to make it one again

end

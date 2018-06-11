function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
m = length(y); % number of training examples
c=alpha/m;     % multiply constant
n = length(theta); %number of feautes including theta 0
delta=zeros(n,1);
hxi_yi=zeros(m,1);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    hxi_yi=X*theta-y;
    for iter1=1:n
      delta(iter1)=c*sum((hxi_yi.*X(:,iter1)));
    end;
    J_history(iter) = computeCost(X, y, theta);
    theta =theta-delta;
end
end

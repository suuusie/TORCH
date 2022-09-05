function [W1, W2, B] = train_TORCH(X1, X2, param, L, S1, S2)
% warning('off');
% fprintf('training...\n');

%% set the parameters
nbits = param.nbits;
eta1 = param.eta1;
eta2 = param.eta2;
eta3 = param.eta3;
eta4 = param.eta4;
eta5 = param.eta5;
gamma = param.gamma;
rho = param.rho;

%% get the dimensions
[n, d1] = size(X1);
d2 = size(X2,2);

%% transpose the matrices
X1 = X1'; X2 = X2'; 
num_class2 = size(L,2) - param.num_class1;
% L(L==0) = -1;
L = L(:, (param.num_class1+1):end);

%% initialization

B = sign(randn(nbits, n));
Zb = sign(randn(nbits, n));
Gb = B - Zb;
V = randn(nbits, num_class2);
R = randn(nbits, num_class2);

%% iterative optimization
for iter = 1:param.iter
    fprintf('iter=%d\n',iter);
    
    % update Y
    Y = inv(eta1 * B * B' + (eta2 + eta3) * V * V' + eta4 * eye(nbits)) * (eta1 * nbits * B * L + eta2 * nbits * V * S1' + eta3 * nbits * V * S2' + eta4 * V);
       
   % update B
    % 2.ALM
    B = sign(2 * nbits * eta1 * Y * L' + 2 * gamma * R * L' - eta1 * Y * Y' * Zb + rho * Zb - Gb);
    
    % update Zb
    Zb = sign(rho * B + Gb - Y * Y' * B);
    
    % update Gb
    Gb = Gb + rho * (B - Zb);
    
    % update V
    Z = eta2 * nbits * Y * S1 + eta3 * nbits * Y * S2 + eta4 * Y;
    Z = Z';
    Temp = Z'*Z-1/num_class2*(Z'*ones(num_class2,1)*(ones(1,num_class2)*Z));
    [~,Lmd,QQ] = svd(Temp); clear Temp
    idx = (diag(Lmd)>1e-6);
    Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
    P = (Z-1/num_class2*ones(num_class2,1)*(ones(1,num_class2)*Z)) *  (Q / (sqrt(Lmd(idx,idx))));
    if num_class2 > nbits-length(find(idx==1))
       P_ = orth(randn(num_class2,nbits-length(find(idx==1))));
    else
       P_ = orth(randn(num_class2,nbits-length(find(idx==1)))')';
    end
    V = sqrt(num_class2)*[P P_]*[Q Q_]';
    V = V';
    
    % update R
    R = B * L * inv(L' * L);

        
end
    % update W1
    W1 = B * X1' * inv(X1 * X1' + eta5 * eye(d1));
      
    % update W2
    W2 = B * X2' * inv(X2 * X2' + eta5 * eye(d2));
    
    
    
    

function update = tik_inv(J,lam,y)
% Regularised tikhonov inversion

% INPUT
% J = Jacobian/measurement sensitivity
% lam = regulariser
% y = residual

% OUTPUT
% update = solution to regualrised tikhonov system 

% generate regularised hessian
alph = max(sum(J.^2));
H = J.'*J;
diagIdx = 1:(size(H,1) + 1):numel(H);
H(diagIdx) = H(diagIdx) + lam*alph;

% solve for update
Jty = J.'*y;
update = H\Jty;
end
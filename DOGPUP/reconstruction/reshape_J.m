function w = reshape_J(J,Js,lam)
% Solve for weightings to reshape J to target

% INPUT
% J = Jacobian/measurement sensitivity
% Js = objective sensitivity to reshape to
% lam = regulariser

% OUTPUT
% w = weighting matrix to apply to LHS of J and residual to form
% sensitivity

A = J*J.';
A = A + lam.*max(diag(A)).*eye(size(A,1));
w = (Js*J.')/(A);
end


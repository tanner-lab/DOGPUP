function [new_mu,iter,err,errPerc,errDel] = dogpup_recon(mesh,data_m,gates,pos_s,lam,type,floor)
% Iteratively reconstructions internal absorption and scattering from boudnary
% measurements
% Based on weighted Levenburg-Marquardt minimisation scheme, weighting is
% computed to reformat Jacobian to gaussian spots

% INPUTS
% mesh = DOGPUP mesh class
% phiM = gated measured data [NM x NG]
% gates = time domain gating functions [NG x NT]
% pos_s = positons of recon basis spots [NW x 3 (x,y,z)] (if empty no weigtings used)
% lam = regulariser for LM minimisation (1) and J reshape (2) [1 x 2]
% type = flag to also reconstruct musp if set to 'full', otherwise only
% absorption is reconstructed

% OUTPUTS
% new_mu = reconstructed absorption values [NN x 1]
% count = exit iteration number [scalar]
% err = absolute residual at each step [count+1 x 1]
% errPerc = % residual at each step [count+1 x 1]
% err = % residual change at each step [count x 1]

% NM = number of measurements, NN = number of nodes, NG = number of gates,
% NT = number of time steps (not gates), NG = number gating functions

if nargin < 6 || isempty(type)
    type = [];
elseif strcmp(type,'full')

else 
    error('type input is either an empty array or ''full''')
end

if nargin < 7 || isempty(floor)
    floor = [];
end

% we dont want to manipulate our orignal mesh so we copy it here
mesh_r = copy(mesh);

% data initialisation
% find data points in 80% thresh on falling edge
id_incl = id_thresh(data_m,0.8,floor);
id_incl = id_incl(:);
% flatten data
data_m = log(data_m(:));
data_m = data_m(id_incl);
% convert time gates to fourier coefficients
gates = td2fc(gates,mesh_r.optode.fAxis,mesh_r.optode.tAxis,2);

% recon basis target spots
if ~isempty(pos_s)
    spots = weighting_spots(mesh_r,pos_s,mesh_r.dxyz/2,'gauss');
    if strcmp(type,'full')
        spots = cat(1,[spots zeros(size(spots))],[zeros(size(spots)) -spots]);
    end
end

% loop initialisation
k = 10^(1/4); % lambda multiplier
iter = 0; % iteration count
maxIter = 10; % max iterations
err = []; % absolute error
errPerc = []; % percentage error
errDel = []; % change in percentage error
w = 1; % weights

% Weighted LM minimisation ------------------------------------------------

fprintf('\n=======================================\n')
fprintf('============= iteration %d =============\n',iter(end))


while 1
    % generate complex FD data and Jacobian
    fprintf('generating forward data and jacobian...')

    [J,data_c] = mesh_r.J_complex([],[],type,false);
    % convert to time gates and flatten
    data_c = fc2tg(data_c,gates,mesh_r.optode.fAxis,numel(mesh_r.optode.tAxis),2);
    J = fc2tg(J,gates,mesh_r.optode.fAxis,numel(mesh_r.optode.tAxis),2)./data_c; % rytov normalisation
    data_c = log(data_c(:));
    data_c = data_c(id_incl);
    J = reshape(J,[],size(J,3));
    J = J(id_incl,:);
    % model-data misfit
    y = data_m - data_c;
   
    fprintf('done!\n')

    % weight Jacobian and error
    if ~isempty(pos_s) && length(lam) == 2 && ~isempty(lam(2))
        fprintf('weighting jacobian and residuals...')
        w = reshape_J(J,spots,lam(2));
        fprintf('done!\n')
    end
    
    J = w*J;
    y = w*y;

    % assume we have decoupled properties
    if strcmp(type,'full')
        J(1:end/2,end/2+1:end) = 0;
        J(end/2+1:end,1:end/2) = 0;
    end

    % error display
    err = [err norm(y)]; % absolute error
    errPerc = [errPerc err(end)/norm(w*data_m)*100]; % percentage error
    fprintf('error: %2.3f%%\n',errPerc(end))
    if iter > 0
        errDel = [errDel (err(end) - err(end-1))/err(end-1)*100];
        fprintf('error change: %2.3f%%\n',errDel(end))
    end

    % convergence check
    if iter > 1 && sign(errDel(end)) == -1 && (abs(errDel(end)) < 2 || errPerc(end) < 0.5)
        fprintf('\nconvergence reached @ iteration %d!\n\n',iter)
        break
    end

    % initialise optical proprty vectors
    if strcmp(type,'full')
        mu = [mesh_r.mua mesh_r.musp];
    else
        mu = mesh_r.mua;
    end
     

    iter = iter + 1;
    fprintf('\n============= iteration %d =============\n',iter(end))


    % Update regulariser depending on change in error
    if iter(end) == 1 || sign(errDel(end)) == -1
        % if error reduces continue with algorithm
    else
        % if error increases step back to previous iteration values and
        % increase regulariser
        fprintf('error increased! previous values used and regularisation increased\n')
        mu = mu - d_mu;
        y = y_prev;
        J = J_prev;
        lam(1) = lam(1)*k; % update lambda
    end

    % threshold regulariser based on absolute error
    if lam(1) < err(end)
        lam(1) = err(end);
    end
    fprintf('lambda: %1.2e\n',lam(1))
    fprintf('solving for optical property update...')

    % solve for optical property update
    if strcmp(type,'full') % solve only for absorption and scattering
        update = [tik_inv(J(1:end/2,1:end/2),lam(1),y(1:end/2));...
                tik_inv(J(end/2+1:end,end/2+1:end),lam(1),y(end/2+1:end))];
    else % solve only for absorption
        update = tik_inv(J,lam(1),y);
    end
    d_mu = mesh_r.g2m*(reshape(update,size(mesh_r.m2g,1),[])); % interpolate to mesh

    % NaN exit condition
    if any(isnan(d_mu))
        warning('NaN calculated... stopping reconstruction')
        fprintf('\n')
        break
    end

    new_mu = mu + d_mu; % update properties
    fprintf('done!\n')

    % non-physical value exit condition
    if any(new_mu < 0)
        new_mu = new_mu - d_mu;
        warning('Negative properties calculated... stopping reconstruction')
        fprintf('\n')
        break
    end

    % update DOGPUP mesh
    if strcmp(type,'full') 
        mesh_r.update_properties(new_mu);
    else
        mesh_r.update_properties([new_mu mesh_r.musp]);
    end
    fprintf('optical properties updated\n')

     if iter == maxIter
        break
     end

    % cache previous values
    y_prev = y;
    J_prev = J;

end
end


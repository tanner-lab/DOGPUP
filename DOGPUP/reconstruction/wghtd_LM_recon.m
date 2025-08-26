function [new_mu,count,err,errPerc,errDel] = wghtd_LM_recon(mesh,data_m,gates,pos_s,lam)
% Iteratively reconstructions internal absorption from boudnary
% measurements
% Based on weighted Levenburg-Marquardt minimisation scheme, weighting is
% computed to reformat Jacobian to gaussian spots

% INPUTS
% mesh = DOGPUP mesh class
% phiM = gated measured data [NM x NG]
% gates = time domain gating functions [NG x NT]
% pos_s = positons of recon basis spots [NW x 3 (x,y,z)] (if empty no weigtings used)
% lam = regulariser for LM minimisation (1) and J reshape (2) [1 x 2]

% OUTPUTS
% new_mu = reconstructed absorption values [NN x 1]
% count = exit iteration number [scalar]
% err = absolute residual at each step [count+1 x 1]
% errPerc = % residual at each step [count+1 x 1]
% err = % residual change at each step [count x 1]

% NM = number of measurements, NN = number of nodes, NG = number of gates,
% NT = number of time steps (not gates), NG = number gating functions

% data initialisation
% find data points in 80% thresh on falling edge
id_incl = id_thresh(data_m,0.8);
id_incl = id_incl(:);
% flatten data
data_m = log(data_m(:));
data_m = data_m(id_incl);
% convert time gates to fourier coefficients
gates = td2fc(gates,mesh.optode.fAxis,mesh.optode.tAxis,2);

% recon basis target spots
if ~isempty(pos_s)
    spots = target_spots(mesh,pos_s,1.5,'gauss'); 
end

% loop initialisation
k = 10^(1/4); % lambda multiplier
lam(1) = lam(1)*k; % update lambda
count = 0; % iteration count
maxIter = 10; % maximum number of iterations
err = []; % absolute error
errPerc = []; % percentage error
errDel = []; % change in percentage error

% Weighted LM minimisation ------------------------------------------------

fprintf('\n=======================================\n')
fprintf('============= iteration %d =============\n',count(end))


while 1
    % generate complex FD data and Jacobian
    fprintf('generating forward data and jacobian... ')

    [J,data_c] = J_complex(mesh,[],[],false);
    % convert to time gates and flatten
    data_c = fc2tg(data_c,gates,mesh.optode.fAxis,numel(mesh.optode.tAxis),2);
    J = fc2tg(J,gates,mesh.optode.fAxis,numel(mesh.optode.tAxis),2)./data_c; % rytov normalisation
    data_c = log(data_c(:));
    data_c = data_c(id_incl);
    J = reshape(J,[],size(J,3));
    J = J(id_incl,:);
    % model-data misfit
    y = data_m - data_c;

    fprintf('done!\n')

    if ~isempty(pos_s)
    % weight Jacobian and error
        fprintf('weighting jacobian and residuals...')
        w = reshape_J(J,spots,lam(2));
        J = w*J;
        y = w*y;
        fprintf('done!\n')
    else
        w = 1;
        J = w*J;
        y = w*y;
    end

    % error display
    err = [err norm(y)]; % absolute error
    errPerc = [errPerc err(end)/norm(w*data_m)*100]; % percentage error
    fprintf('error: %2.3f%%\n',errPerc(end))
    if count > 0
        errDel = [errDel errPerc(end) - errPerc(end-1)];
        fprintf('error change: %2.3f%%\n',errDel(end))
    end

    % convergence check
    if count > 1 && sign(errDel(end)) == -1 && (abs(errDel(end)) < 0.12 || errPerc(end) < 1.5)
        fprintf('\nconvergence reached @ iteration %d!\n\n',count)
        break
    end

    count = count + 1;
    fprintf('\n============= iteration %d =============\n',count(end))

    % Update regulariser depending on change in error
    if count(end) == 1 || sign(errDel(end)) == -1
        % if error reduces continue with algorithm and reduce regulariser
        G = mesh.m2g*mesh.mua;
        J = J.*G.'; % property normalisation
        lam(1) = lam(1)/k; % update lambda
    else
        % if error increases step back to previous iteration values and
        % increase regulariser
        fprintf('error increased! previous values used and regularisation increased\n')
        G = mesh.m2g*(mesh.mua - d_mu);
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

    
    % update mua based on regualrised inversion
    % solve only for voxels within mesh
    update = tik_inv(J(:,mesh.gridinmesh),lam(1),y);
    % map update to full voxel grid
    x = zeros(prod(mesh.gridSize),1);
    x(mesh.gridinmesh) = update;

    x = G.*x; % rescale update
    d_mu = mesh.g2m*x; % interpolate to mesh

    % NaN exit condition
    if any(isnan(d_mu))
        warning('NaN mua calculated... stopping reconstruction')
        fprintf('\n')
        break
    end

    new_mu = mesh.mua + d_mu; % update absorption
    fprintf('done!\n')

    % non-physical value exit condition
    if any(new_mu < 0)
        new_mu = new_mu - d_mu;
        warning('Negative mua calculated... stopping reconstruction')
        fprintf('\n')
        break
    end

    % update DOGPUP mesh
    mesh = update_properties(mesh,[new_mu mesh.musp]);
    fprintf('optical properties updated\n')

    % max iter exit condtion
    if count > maxIter-1
        fprintf('\nmaximum iteration reached!\n\n')
        break
    end

    % cache previous values
    y_prev = y;
    J_prev = J;

end
end


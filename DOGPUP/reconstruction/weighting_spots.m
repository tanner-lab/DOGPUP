function spots = weighting_spots(mesh,pos,w,type)
    % Get array of sensitivity spots to form when weighting J
    
    % INPUT
    % mesh = fully initialised DOGPUP mesh
    % pos = barycentre of spots [NS x 3] (mm)
    % w = width of spot either FWHM ('gauss') or true width ('square)
    % type = 'gauss' or 'square' string, determines it spot is
    % square pixel or gaussian

    % OUTPUT
    % target = array of sensitivity spots (NS x NV)

    % NS = number of spots, NV = number of voxels

    pos = pos.';
    if strcmp(type,'gauss')
        s = (2*sqrt(2*log(2)))*w;
        spots = exp(-(mesh.grid(:,1)-pos(1,:)).^2./(s^2.*2)...
            -(mesh.grid(:,2)-pos(2,:)).^2./(s^2.*2)...
            -(mesh.grid(:,3)-pos(3,:)).^2./(s^2.*2));
    elseif strcmp(type,'square')
        w = w/2;
        spots = double(mesh.grid(:,1) <= (pos(1,:) + w) & mesh.grid(:,1) >= (pos(1,:) - w)...
            & mesh.grid(:,2) <= (pos(2,:) + w) & mesh.grid(:,2) >= (pos(2,:) - w)...
            & mesh.grid(:,3) <= (pos(3,:) + w) & mesh.grid(:,3) >= (pos(3,:) - w));
    else 
        error('Check spot type is gauss or square')
    end
    spots = spots./sum(spots,1);
    spots = -(spots).';
    spots = spots(:,mesh.gridinmesh);
end
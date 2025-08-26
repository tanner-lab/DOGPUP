classdef dMesh
    % DOGPUP mesh class

    properties (Access = public)
        % MESH GEOMETRY
        node % mesh nodes
        elem % mesh tetrehedral elements
        vol % mesh tetra volumes
        face % mesh face triangle elements
        area % mesh triangle areas
        bnd % binary boundary marker
        gradScale % scaling matrix for gradient basis functions

        % MESH PROPERTIES
        nr % global refractive index
        c % speed of light in media
        mua % absorption coeff mm^-1
        musp % reduced scattering coeff mm^-1
        kappa % diffusion coeff mm
        R % boundary factor

        % Matrices for FEM
        K % diffusion matrix
        M % mass matrix

        % Optodes
        optode % DOGPUP optode class handle

        % GRID
        grid % grid co-ordinates
        gridSize % [x y z] num voxels
        dxyz % voxel volume

        % INTERPOLATION MATRICES
        m2g % mesh to grid matrix
        g2m % grid to mesh matrix
        gridinmesh % non-zero grid points
        
    end

    properties (Constant)
        cVac = 2.99792458e11; % speed of light in vaccum (mms^-1)
    end
    
    methods 
        %% Mesh Setup
        % Construct an instance of this class
        function mesh = dMesh(node,elem,nr,props)
            % Construct DOGPUP mesh from given mesh geometry and optical
            % properties

            % INPUT
            % node = node locations [NN x 3] (mm)
            % elem = mesh connectivity list [NE x 4]
            % nr = refractive index of mesh [scalar]
            % props = optical properties of mesh [mua musp], either [1 x 2] or [NN x 2]

            % mua = absoprtion (mm^-1), musp = reduced scattering (mm^-1)

            % OUTPUT
            % mesh = DOGPUP mesh class

            % NN = number of nodes, NE = number of elements, 

            mesh.node = node;
            mesh.elem = elem;
            mesh.nr = nr;
            mesh.mua = props(:,1);
            mesh.musp = props(:,2);
            if size(props,1) == 1
                mesh.mua = ones(size(node,1),1).*mesh.mua;
                mesh.musp = ones(size(node,1),1).*mesh.musp;
            end
            mesh.kappa = 1./(3.*(mesh.mua + mesh.musp));
            mesh = prepare_mesh(mesh); % prepare geometry
            mesh = update_properties(mesh); % generate forward matrix
        end

        % update optical properties and generate forward matrix
        function mesh = update_properties(mesh,props)
            % Generates forward matrix for FEM solution of fluence
            if nargin > 1
                mesh.mua(:)= props(:,1);
                mesh.musp(:) = props(:,2);
                mesh.kappa(:) = 1./(3.*(mesh.mua + mesh.musp));
            end
            [mesh.K,mesh.M] = gen_fwdmat(mesh);
        end

        % link optode class to mesh class
        function mesh = add_optode(mesh,optode)
            % snaps optodes to nearese surface of the mesh and links the
            % two objects
            optode = snap2mesh(optode,mesh);
            mesh.optode = optode;
        end
        
        % clear optodes
        function mesh = clear_optode(mesh)
            % unlinks optodes
            mesh.optode = [];
        end
    
        %% Fluence solving methods
        % solves for forward fluence using BICGSTAB and FSAI preconditioning
        function [phi,data] = flu_solve(mesh,displayFlag)
            % Finds FEM solution for fluence and data at detectors

            % INPUT
            % mesh = fully initialised DOGPUP mesh
            % displayFlag = flag for text display (boolean)
            
            % OUTPUT
            % phi = Fourier coefficients of fluence through mesh (NN x NF x NS)
            % data = Fourier coefficients of fluence at detectors (NM x NF)

            % NN = number of nodes, NF = number of frequencies, 
            % NS = number of sources, NM = number of measurements

            if nargin < 2
                displayFlag = 1;
            end

            % generate source vectors
            Q = zeros(size(mesh.node,1),1,size(mesh.optode.s_positions,1));
            for i = 1:size(mesh.optode.s_positions,1)
                Q(mesh.elem(mesh.optode.s_bary(i,1),:),:,i) = mesh.optode.s_bary(i,2:end);
            end
            Q = Q.*mesh.optode.s_fpsf;
            % solve
            tdisp('\nsolving for fluence...',displayFlag)
            phi = fluGPU(mesh,Q);
            tdisp('done!\n',displayFlag)

            % get detector data
            if nargout > 1
                data = meas_flu(mesh,phi);
                tdisp('detection generated!\n',displayFlag)
            end

        end

        % gets data from detectors
        function data = meas_flu(mesh,phi)
            % Finds fluence at detectors
            
            % INPUT
            % mesh = fully initialised DOGPUP mesh
            % phi = Fourier coefficients of fluence through mesh (NN x NF x NS)

            % OUTPUT
            % data = Fourier coefficients of fluence at detectors (NM x NF)

            i = kron((1:size(mesh.optode.d_positions,1)).',ones(3,1));
            j = gather(mesh.elem(mesh.optode.d_bary(:,1),:)).';
            j = double(j(:));
            j = j(mesh.optode.d_bary(:,2:end).'>0);
            v = mesh.optode.d_bary(:,2:end).';
            v = v(:);
            v = v(mesh.optode.d_bary(:,2:end).'>0);
            % Detection matrix
            D = sparse(i,j,v,size(mesh.optode.d_bary,1),gather(size(mesh.node,1)));

            N = size(phi,2);
            data = zeros(size(mesh.optode.link,1),N);
            idx = sub2ind([size(D,1) size(phi,3)],mesh.optode.link(:,2),mesh.optode.link(:,1));
            for i = 1:N
                temp = D*squeeze(phi(:,i,:));
                data(:,i) = temp(idx);
            end
        end
        
        % solves for adjoint fluence using BICGSTAB and FSAI preconditioning
        function phiA = adj_flu_solve(mesh,displayFlag)
            % Finds FEM solution for adjoint fluence

            % INPUT
            % mesh = fully initialised DOGPUP mesh
            % displayFlag = flag for text display (boolean)
            
            % OUTPUT
            % phiA = Fourier coefficients of adjoint fluence through mesh (NN x NF x ND)

            % NN = number of nodes, NF = number of frequencies, 
            % ND = number of detectors

            if nargin < 2
                displayFlag = 1;
            end

            % generate source vectors
            Q = zeros(size(mesh.node,1),1,size(mesh.optode.d_positions,1));
            for i = 1:size(mesh.optode.d_positions,1)
                Q(mesh.elem(mesh.optode.d_bary(i,1),:),:,i) = mesh.optode.d_bary(i,2:end);
            end
            Q = Q.*mesh.optode.s_fpsf;
            % solve
            tdisp('\nsolving for adjoint fluence...',displayFlag)
            phiA = fluGPU(mesh,Q);
            tdisp('done!\n',displayFlag)

        end

        %% Inverse probelm methods
        % generates complex FD absorption jacobian
        function [J,data,phi,phiA] = J_complex(mesh,phi,phiA,displayFlag)
            % Finds the Fourier series coefficient absorption Jacobian 
            % using the adjoint method

            % INPUT
            % mesh = fully initialised DOGPUP mesh
            % displayFlag = flag for text display (boolean)
            
            % OUTPUT
            % J = absorption sensitivity for Fourier coefficients (NM x NF x NV)
            % data = Fourier coefficients of fluence at detectors (NM x NF)
            % phi = Fourier coefficients of fluence through mesh (NN x NF x NS)
            % phiA = Fourier coefficients of adjoint fluence through mesh (NN x NF x ND)

            % NN = number of nodes, NF = number of frequencies, 
            % NV = number of voxels, NS = number of sources, 
            % ND = number of detectors, NM = number of measurements

            if nargin < 4
                displayFlag = 1;
            end
            
            if nargin < 3 || isempty(phiA)
                % generate adjoint fluence if not given
                phiA = adj_flu_solve(mesh,displayFlag);
                if nargin < 2 || isempty(phi)
                    % generate forward fluence if not given
                    [phi,data] = flu_solve(mesh,displayFlag);
                end
            end
            
            tdisp('\ngenerating complex absorption jacobian...',displayFlag)
            % interpolate fluences to grid
            phi_Grid = reshape(mesh.m2g*reshape(phi,size(phi,1),[]),[],size(phi,2), size(phi,3));
            phiA_Grid = reshape(mesh.m2g*reshape(phiA,size(phiA,1),[]),[],size(phiA,2), size(phiA,3));
            % convolve and correctly reshape to grid
            J = -phi_Grid(:,:,mesh.optode.link(:,1)).*phiA_Grid(:,:,mesh.optode.link(:,2)).*mesh.dxyz^3; % grid x freq x meas

            J = permute(J,[3 2 1]); % meas x freq x [voxel_kappa voxel_mua]
            J = J./mesh.optode.s_fpsf; % scale convolution
            tdisp('done!\n',displayFlag)
        end

        % Form array of target spots
        function target = target_spots(mesh,pos,w,type)
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
                target = exp(-(mesh.grid(:,1)-pos(1,:)).^2./(s^2.*2)...
                    -(mesh.grid(:,2)-pos(2,:)).^2./(s^2.*2)...
                    -(mesh.grid(:,3)-pos(3,:)).^2./(s^2.*2));
            elseif strcmp(type,'square')
                w = w/2;
                target = double(mesh.grid(:,1) <= (pos(1,:) + w) & mesh.grid(:,1) >= (pos(1,:) - w)...
                    & mesh.grid(:,2) <= (pos(2,:) + w) & mesh.grid(:,2) >= (pos(2,:) - w)...
                    & mesh.grid(:,3) <= (pos(3,:) + w) & mesh.grid(:,3) >= (pos(3,:) - w));
            else 
                error('Check spot type is gauss or square')
            end
            target = target./sum(target,1);
            target = -(target).';
            target(:,~mesh.gridinmesh) = 0;
        end

        %% Slicing and interpolation methods
        % Grid interpolation for reconstruction basis
        function mesh = mesh2grid(mesh,x,y,z)
        % Generates transformation matrices for interpolation to voxel
        % grid and vice versa

        % INPUT
        % mesh = DOGPUP mesh
        % [x,y,z] = grid points [NV x 3] (mm)

        % OUTPUT
        % mesh.grid = grid points [NV x 3] (mm)
        % mesh.gridSize = number of voxels in x,y and z [1 x 3]
        % mesh.dxyz = resolution of voxel grid
        % mesh.m2g = interpolation matrix from mesh to grid [NV x NN]
        % mesh.g2m = interpolation matrix from grid to mesh [NN x NV]
        % mesh.gridinmesh = binary flag, true if voxel is inside mesh [NV x 1]

        % NV = number of voxels, NN = number of nodes
        
        % assumes cubic grid

        % define grid
        dx = abs(x(2) - x(1));
        [X,Y,Z] = ndgrid(x,y,z);
        mesh.grid = [X(:),Y(:),Z(:)];
        
        %% Calculate mesh to grid transform matrix
        % find barycentric co-ords of each voxelised point w.r.t mesh
        TR = triangulation(double(mesh.elem),mesh.node);
        [idInt,bary] = pointLocation(TR,mesh.grid);
        mesh.gridinmesh = logical(~isnan(idInt));
        
        % use barycentric co-ords to form interpolation matrix
        i = find(mesh.gridinmesh);
        i = repelem(i,4,1);
        
        j = double(mesh.elem(idInt(mesh.gridinmesh),:)).';
        j = j(:);
        
        v = bary(mesh.gridinmesh,:).';
        v = v(:);
        
        mesh.m2g = sparse(i,j,v,size(mesh.grid,1),size(mesh.node,1));
        clearvars i j v
        
        %% Calculate grid to mesh transform matrix
        % find barycentric co-ords of each node w.r.t voxel grid
        elemG = delaunay(mesh.grid);
        TR = triangulation(elemG,mesh.grid);
        [idInt,bary] = pointLocation(TR,mesh.node);
        
        % use barycentric co-ords to form interpolation matrix
        % removes weighting from voxel points outside mesh
        
        j = double(elemG(idInt,:)).';
        eleminmesh = logical(mesh.gridinmesh(j));
        j = j(eleminmesh);
        
        i = 1:size(mesh.node,1);
        i = repelem(i,4,1);
        i = i(eleminmesh);
        
        bary = bary.';
        v = bary(eleminmesh);
        
        % find nodes that are outside voxel grid
        if any(sum(eleminmesh,1) < 1)
        
            % nearest neighbour interpolation for nodes that lie in the zero space
            idx = find(sum(eleminmesh,1) < 1);
            iS = 1:size(mesh.node,1);
            iS = iS(idx).';
            elemG = delaunay(mesh.grid(mesh.gridinmesh,:));
            TR = triangulation(elemG,mesh.grid(mesh.gridinmesh,:));
            jS = nearestNeighbor(TR,mesh.node(idx,:));
            jtemp = find(mesh.gridinmesh);
            jS = jtemp(jS);
            vS = ones(length(iS),1);
            
            i = [i; iS];
            j = [j; jS];
            v = [v; vS];
        
        end
        
        mesh.g2m = sparse(i,j,v,size(mesh.node,1),size(mesh.grid,1));
        norm = sum(mesh.g2m,2);
        norm(norm==0) = 1;
        mesh.g2m = mesh.g2m./norm;
        clearvars i j v iS jS vS norm
        
        mesh.gridSize = [length(x), length(y), length(z)];
        mesh.dxyz = dx;
        
        
        end

        % function to interpolate values to different mesh
        function val_int = mesh2mesh(mesh,val,old_node,old_elem)
            % interplotes values between two mesh

            % INPUT
            % mesh = target DOGPUP mesh
            % val = function to be interpolated
            % old_node = nodes of mesh val is defined on
            % old_elem = elements of mesh val is defined on

            % OUTPUT
            % val_int = interpolated value

            % interpolate
            TR = triangulation(gather(double(old_elem)),gather(old_node));
            [ind,bary] = pointLocation(TR,mesh.node);
            val_int = zeros(size(mesh.node,1),size(val,2));
            for i = 1:size(val_int,2)
                temp = val(:,i);
                val_int(:,i) = sum(bary.*temp(old_elem(ind,:)),2);
            end

            % fill NaNs with nearest neighbours
            out_node = mesh.node(isnan(ind));
            in_node = mesh.node(~isnan(out_node));
            k = find(~isnan(out_node));
            k = k(dsearchn(in_node,out_node));
            val_int(out_node,:) = val_int(k,:);

        end
        
        % function to slice mesh to grid for display forms 128 x 128 image
        function [sliceMat,points] = mesh_slice(mesh,plane)
            % generates matrix transform to find slice of function defined
            % on mesh

            % INPUT
            % mesh = DOGPUP mesh
            % plane = string that is formated 'x=plane', 'y=plane' or
            % 'z=plane', determines slicing plance

            % OUTPUT
            % sliceMat = matrix to interpolate at slice [256*256 x NN]
            % points = (x,y,z) points on slice [256*256 x 3]

            % NN = number of nodes
            
            if ischar(plane) == 0
                error('Char input required for slicing plane')
            elseif ~strcmpi(plane(1:2),'x=') && ~strcmpi(plane(1:2),'y=') && ~strcmpi(plane(1:2),'z=')
                error('Format for slicing plane is similar to ''x=30''')
            end

            res = 256;

            [minPos,maxPos] = bounds(mesh.node,'all');
            x = linspace(minPos,maxPos,res);
            
            if strcmp(plane(1:2),'x=')
                [X,Y,Z] = meshgrid(str2double(extractAfter(plane,'=')),x,x);
            elseif strcmpi(plane(1:2),'y=')
                [X,Y,Z] = meshgrid(x,str2double(extractAfter(plane,'=')),x);
            elseif strcmpi(plane(1:2),'z=')
                [X,Y,Z] = meshgrid(x,x,str2double(extractAfter(plane,'=')));
            end
            
            points = [X(:),Y(:),Z(:)];

            % generate interpolation matrix 
            TR = triangulation(double(mesh.elem),mesh.node);
            [idInt,bary] = pointLocation(TR,points);
            in_idx = logical(~isnan(idInt));
            
            % use barycentric co-ords to form interpolation matrix
            i = find(in_idx);
            i = repelem(i,4,1);
            
            j = double(mesh.elem(idInt(in_idx),:)).';
            j = j(:);
            
            v = bary(in_idx,:).';
            v = v(:);

            iNaN = find(~in_idx);
            jNaN = iNaN;
            jNaN(jNaN>size(mesh.node,1)) = size(mesh.node,1);
            vNaN = NaN.*ones(size(iNaN));
            
            sliceMat = sparse([i;iNaN],[j;jNaN],[v;vNaN],size(points,1),size(mesh.node,1));

        end

        % function to interpolate mesh to 3D grid for display forms 64 x 64 x 64
        % volume
        function [volMat,points] = mesh_slice3(mesh)
            % generates matrix transform to convert data to 64 x 64 x 64
            % grid for display purposes

            % INPUT
            % mesh = DOGPUP mesh

            % OUTPUT
            % volMat = matrix to interpolate to grid [64*64*64 x NN]
            % points = (x,y,z) points in volume [64*64*64 x 3]

            % NN = number of nodes
            
            res = 64;

            [minPos,maxPos] = bounds(mesh.node,'all');
            x = linspace(minPos,maxPos,res-2);
            dx = x(2) - x(1);
            x = cat(2,x(1)-dx,x,x(end)+dx);
            [X,Y,Z] = meshgrid(x,x,x);
            
            points = [X(:),Y(:),Z(:)];

            % generate interpolation matrix 
            TR = triangulation(double(mesh.elem),mesh.node);
            [idInt,bary] = pointLocation(TR,points);
            in_idx = logical(~isnan(idInt));
            
            % use barycentric co-ords to form interpolation matrix
            i = find(in_idx);
            i = repelem(i,4,1);
            
            j = double(mesh.elem(idInt(in_idx),:)).';
            j = j(:);
            
            v = bary(in_idx,:).';
            v = v(:);
            
            volMat = sparse(i,j,v,size(points,1),size(mesh.node,1));

        end
    
        %% Plotting methods
        % plot surface of mesh
        function plotdmesh(mesh,alpha)
            % plots surface mesh with face transparency given by alpha

            % INPUT
            % mesh = DOGPUP mesh
            % alpha = alpha transparency 0 to 1 [scalar]
            
            % plotting limits
            [minPos,maxPos] = bounds(mesh.node,'all');
            len = (1.2*maxPos - 1.2*minPos)/2;
            c0 = mean(mesh.node,1);
            % plot
            p0 = trimesh(mesh.face,mesh.node(:,3),mesh.node(:,1),mesh.node(:,2),'EdgeColor',[0.65 0.65 0.65],'FaceColor',[0.8 0.8 0.8],'FaceAlpha',alpha,'EdgeAlpha',alpha);
            xlim(c0(3) + [-len len])
            xlim(c0(1) + [-len len])
            xlim(c0(2) + [-len len])

            % Format datatip
            mytip = datatip(p0);
            txt1 = dataTipTextRow('X','YData');
            txt2 = dataTipTextRow('Y','ZData');
            txt3 = dataTipTextRow('Z','XData');
            p0.DataTipTemplate.DataTipRows(1) = txt1;
            p0.DataTipTemplate.DataTipRows(2) = txt2;
            p0.DataTipTemplate.DataTipRows(3) = txt3;
            delete(mytip);

            % view, labels and scale
            xlabel('z (mm)')
            ylabel('x (mm)')
            zlabel('y (mm)')
            view(45,45)
            axis equal
            drawnow
        end
        
        % plot surface with optodes
        function plotdmesh_snd(mesh,alpha,lbl_flag)
            % plots surface mesh with face transparency given by alpha and
            % optodes

            % INPUT
            % mesh = DOGPUP mesh
            % alpha = alpha transparency 0 to 1 [scalar]
            % lbl_flag = boolean flag to show optode numbering, default
            % true

            % plot
            hold on
            plotdmesh(mesh,alpha)
            scatter3(mesh.optode.s_positions(:,3),mesh.optode.s_positions(:,1),mesh.optode.s_positions(:,2),20,'r','filled')
            scatter3(mesh.optode.d_positions(:,3),mesh.optode.d_positions(:,1),mesh.optode.d_positions(:,2),20,'b','filled')

            if nargin < 3 || lbl_flag == true
                % number sources and detectors
                stxt = strsplit(num2str(1:size(mesh.optode.s_dirs,1))); % source labels
                dr = -3.*mesh.optode.s_dirs;
                txtPos = mesh.optode.s_positions;
                text(txtPos(:,3)+dr(:,3),txtPos(:,1)+dr(:,1),txtPos(:,2)+dr(:,2),stxt,'Color','red')
    
                dtxt = strsplit(num2str(1:size(mesh.optode.d_dirs,1)));
                dr = -3.*mesh.optode.d_dirs;
                txtPos = mesh.optode.d_positions;
                text(txtPos(:,3)+dr(:,3),txtPos(:,1)+dr(:,1),txtPos(:,2)+dr(:,2),dtxt,'Color','blue')
            end

            % view, labels and scale
            xlabel('z (mm)')
            ylabel('x (mm)')
            zlabel('y (mm)')
            view(-45,45)
            daspect([1 1 1])
            drawnow

        end
    
        % plot slice through mesh, interpoalted to 128 x 128 grid
        function plotfun_slice(mesh,fun,map,plane,incl)
            % Plots 2D slice of function defined on mesh

            % plots surface mesh with face transparency given by alpha

            % INPUT
            % mesh = DOGPUP mesh
            % fun = function defined on mesh [NN x 1]
            % map = matlab colormap
            % plane = string that is formated 'x=plane', 'y=plane' or
            % 'z=plane', determines slicing plance
            % incl = inclusion array [NI X (x,y,z) radius]

            % NN = number of nodes, NI = number of inclusions

            % slice function on plane
            [sliceMat,points] = mesh_slice(mesh,plane);
            fun = sliceMat*fun;

            % plot inclusion stats
            if nargin > 4
                r = incl(:,4);
                c0 = incl(:,1:3);
            end
            
            res = 256;
            cc = colororder;
            cc = cc(1,:);

            if strcmp(plane(1:2),'x=')
                % plot sliced plane
                y = reshape(points(:,2),res,res);
                z = reshape(points(:,3),res,res);
                fun = reshape(fun,res,res);
                colormap(map)
                im = imagesc([z(1) z(end)],[y(1) y(end)],fun);
                set(im, 'AlphaData', ~isnan(fun))
                set(gca,'YDir','normal')
                xlabel('z (mm)')
                ylabel('y (mm)')
                % format datatip
                mytip = datatip(im);
                txt1 = dataTipTextRow('[Z,Y]','[X,Y]');
                txt2 = dataTipTextRow('Val','Index');
                im.DataTipTemplate.DataTipRows(1) = txt1;
                im.DataTipTemplate.DataTipRows(2) = txt2;
                im.DataTipTemplate.DataTipRows(3) = [];
                delete(mytip);
                xlim([floor(min(z(~isnan(fun)))) ceil(max(z(~isnan(fun))))])
                ylim([floor(min(y(~isnan(fun)))) ceil(max(y(~isnan(fun))))])

                % plot inclusion ground truth
                if  nargin > 4
                    for i = 1:length(r)
                        circle_r2 = r(i)^2 - (points(1,1) - c0(i,1)).^2;
                        if circle_r2 < 0
                            continue
                        end
                        circle_r = sqrt(circle_r2);
                        theta = linspace(0, 2*pi, 100);
                        y = c0(i,2) + circle_r.*cos(theta);
                        z = c0(i,3) + circle_r.*sin(theta);
                        hold on
                        plot(z,y,'LineWidth',1.5,'Color',cc)
                        hold off
                    end
                end

            elseif strcmpi(plane(1:2),'y=')
                % plot sliced plane
                x = reshape(points(:,1),res,res);
                z = reshape(points(:,3),res,res);
                fun = reshape(fun,res,res).';
                colormap(map)
                im = imagesc([x(1) x(end)],[z(1) z(end)],fun);
                set(im, 'AlphaData', ~isnan(fun))
                set(gca,'YDir','normal')
                xlabel('x (mm)')
                ylabel('z (mm)')
                % format datatip
                mytip = datatip(im);
                txt1 = dataTipTextRow('[X,Z]','[X,Y]');
                txt2 = dataTipTextRow('Val','Index');
                im.DataTipTemplate.DataTipRows(1) = txt1;
                im.DataTipTemplate.DataTipRows(2) = txt2;
                im.DataTipTemplate.DataTipRows(3) = [];
                delete(mytip);
                xlim([floor(min(x(~isnan(fun)))) ceil(max(x(~isnan(fun))))])
                ylim([floor(min(z(~isnan(fun)))) ceil(max(z(~isnan(fun))))])

                % plot inclusion ground truth
                if  nargin > 4
                    for i = 1:length(r)
                        circle_r2 = r(i)^2 - (points(1,2) - c0(i,2)).^2;
                        if circle_r2 < 0
                            continue
                        end
                        circle_r = sqrt(circle_r2);
                        theta = linspace(0, 2*pi, 100);
                        x = c0(i,1) + circle_r.*cos(theta);
                        z = c0(i,3) + circle_r.*sin(theta);
                        hold on
                        plot(x,z,'LineWidth',1.5,'Color',cc)
                        hold off
                    end
                end

            elseif strcmpi(plane(1:2),'z=')
                % plot sliced plane
                x = reshape(points(:,1),res,res);
                y = reshape(points(:,2),res,res);
                fun = reshape(fun,res,res);
                colormap(map)
                im = imagesc([y(1) y(end)],[x(1) x(end)],fun);
                set(im, 'AlphaData', ~isnan(fun))
                set(gca,'YDir','normal')
                % view(0,270)
                xlabel('x (mm)')
                ylabel('y (mm)')
                % format datatip
                mytip = datatip(im);
                txt1 = dataTipTextRow('[X,Y]','[X,Y]');
                txt2 = dataTipTextRow('Val','Index');
                im.DataTipTemplate.DataTipRows(1) = txt1;
                im.DataTipTemplate.DataTipRows(2) = txt2;
                im.DataTipTemplate.DataTipRows(3) = [];
                delete(mytip);
                xlim([floor(min(x(~isnan(fun)))) ceil(max(x(~isnan(fun))))])
                ylim([floor(min(y(~isnan(fun)))) ceil(max(y(~isnan(fun))))])

                % plot inclusion ground truth
                if  nargin > 4
                    for i = 1:length(r)
                        circle_r2 = r(i)^2 - (points(1,3) - c0(i,3)).^2;
                        if circle_r2 < 0
                            continue
                        end
                        circle_r = sqrt(circle_r2);
                        theta = linspace(0, 2*pi, 100);
                        x = c0(i,1) + circle_r.*cos(theta);
                        y = c0(i,2) + circle_r.*sin(theta);
                        hold on
                        plot(x,y,'LineWidth',1.5,'Color',cc)
                        hold off
                    end
                end
            end

            clim([min(fun(:)) max(fun(:))])
            set(gca,'Color',[0.9 0.9 0.9])
            daspect([1 1 1])
            drawnow

        end
    
        % plot slice through reconstruction basis, no interpolation
        function plotfun_grid(mesh,fun,map,plane)
            % plots slice through function defined on voxel grid
            % see plotfun_slice
            
            res = mesh.gridSize;
            s_pos = str2double(extractAfter(plane,'='));
            x = reshape(mesh.grid(:,1),res(1),res(2),res(3));
            y = reshape(mesh.grid(:,2),res(1),res(2),res(3));
            z = reshape(mesh.grid(:,3),res(1),res(2),res(3));
            fun(~mesh.gridinmesh) = NaN;
            fun = reshape(fun,res(1),res(2),res(3));

            if strcmp(plane(1:2),'x=')
                % convert slice position to nearest slice index
                x_s = unique(x(:,1,1));
                [~,sIdx] = min(abs(x_s-s_pos));
                % plot sliced plane
                y = pagetranspose(squeeze(y(sIdx,:,:)));
                z = pagetranspose(squeeze(z(sIdx,:,:)));
                fun = squeeze(fun(sIdx,:,:));
                colormap(map)
                im = imagesc([z(1) z(end)],[y(1) y(end)],fun);
                set(im, 'AlphaData', ~isnan(fun))
                view(0,270)
                xlabel('z (mm)')
                ylabel('y (mm)')
                % format datatip
                mytip = datatip(im);
                txt1 = dataTipTextRow('[Z,Y]','[X,Y]');
                txt2 = dataTipTextRow('Val','Index');
                im.DataTipTemplate.DataTipRows(1) = txt1;
                im.DataTipTemplate.DataTipRows(2) = txt2;
                im.DataTipTemplate.DataTipRows(3) = [];
                delete(mytip);

            elseif strcmpi(plane(1:2),'y=')
                % convert slice position to nearest slice index
                y_s = unique(y(1,:,1));
                [~,sIdx] = min(abs(y_s-s_pos));
                % plot sliced plane
                x = pagetranspose(squeeze(x(:,sIdx,:)));
                z = pagetranspose(squeeze(z(:,sIdx,:)));
                fun = pagetranspose(squeeze(fun(:,sIdx,:)));
                colormap(map)
                im = imagesc([x(1) x(end)],[z(1) z(end)],fun);
                set(im, 'AlphaData', ~isnan(fun))
                view(0,270)
                xlabel('x (mm)')
                ylabel('z (mm)')
                % format datatip
                mytip = datatip(im);
                txt1 = dataTipTextRow('[X,Z]','[X,Y]');
                txt2 = dataTipTextRow('Val','Index');
                im.DataTipTemplate.DataTipRows(1) = txt1;
                im.DataTipTemplate.DataTipRows(2) = txt2;
                im.DataTipTemplate.DataTipRows(3) = [];
                delete(mytip);

            elseif strcmpi(plane(1:2),'z=')
                % convert slice position to nearest slice index
                z_s = unique(z(1,1,:));
                [~,sIdx] = min(abs(z_s-s_pos));
                % plot sliced plane
                x = pagetranspose(squeeze(x(:,:,sIdx)));
                y = pagetranspose(squeeze(y(:,:,sIdx)));
                fun = pagetranspose(squeeze(fun(:,:,sIdx)));
                colormap(map)
                im = imagesc([x(1) x(end)],[y(1) y(end)],fun);
                set(im, 'AlphaData', ~isnan(fun))
                view(0,270)
                xlabel('x (mm)')
                ylabel('y (mm)')
                % format datatip
                mytip = datatip(im);
                txt1 = dataTipTextRow('[X,Y]','[X,Y]');
                txt2 = dataTipTextRow('Val','Index');
                im.DataTipTemplate.DataTipRows(1) = txt1;
                im.DataTipTemplate.DataTipRows(2) = txt2;
                im.DataTipTemplate.DataTipRows(3) = [];
                delete(mytip);               
            end

            set(gca,'Color',[0.9 0.9 0.9])
            daspect([1 1 1])
            drawnow

        end
        
        % plot data along line
        function [fun_l,line] = plotfun_line(mesh,fun,line)
            % Either plots data defined on mesh along input line for just
            % extracts the data from the line

            % INPUTS
            % mesh = DOGPUP mesh
            % fun = function defined on mesh [NN x 1]
            % line = start and end points of line [2 x (x,y,z)]

            % OUTPUTS
            % fun_l = function on line
            % line = points in 3D for each point on line [NP x (x,y,z)]

            % NN = number of nodes, NP = number of points on line
            
            % define line by length and direction
            dir = line(2,:) - line(1,:);
            len = vecnorm(dir,2,2); % line length
            dir = dir./len; % line direction
          
            % line as x y z points
            x = line(1,1):dir(1):line(2,1);
            if isempty(x)
                x = line(1,1);
            end
            y = line(1,2):dir(2):line(2,2);
            if isempty(y)
                y = line(1,2);
            end
            z = line(1,3):dir(3):line(2,3);
            if isempty(z)
                z = line(1,3);
            end
            [x,y,z] = ndgrid(x,y,z);
            line = [x(:),y(:),z(:)];

            % interpolate on to line
            TR = triangulation(double(mesh.elem),mesh.node);
            [idInt,bary] = pointLocation(TR,line);
            fun_l = zeros(size(idInt));
            fun_l(~isnan(idInt)) = sum(bary(~isnan(idInt),:).*fun(mesh.elem(idInt(~isnan(idInt)),:)),2);

            if nargout > 0
            else
                % plot
                d = vecnorm(line - line(1,:),2,2);
                clr = colororder;
                clr = clr(2,:);
                plot(d,fun_l,'Color',clr)
                xlabel('Distance (mm)')
            end

        end
        
        % plot isosurface
        function plotfun_vol(mesh,funGT,fun)
            % plots dice thresholded isosurface of function defined on mesh

            % INPUT
            % mesh = DOGPUP mesh
            % funGT = function defined on mesh [NN x 1]
            % fun = another function defined on mesh [NN x 1]

            % NN = number of nodes

            [sliceMat,points] = mesh_slice3(mesh);
            [points,idx] = sortrows([points(:,3),points(:,1),points(:,2)]);
            Z = reshape(points(:,1),64,64,64);
            X = reshape(points(:,2),64,64,64);
            Y = reshape(points(:,3),64,64,64);

            hold on
            plotdmesh(mesh,0.1)
            if nargin > 2
                fun = sliceMat*fun;
                fun = reshape(fun(idx),64,64,64);
                iso_thresh = 0.5.*(median(fun(fun>0),'all') + max(fun,[],'all'));
                iso = isosurface(Z,X,Y,fun,iso_thresh);
                p = patch(iso);
                set(p,'FaceColor',[1 0 0],'FaceAlpha',0.4);  
                set(p,'EdgeColor','none');
                camlight;
                lighting gouraud;
            end
            
            funGT = sliceMat*funGT;
            funGT = reshape(funGT(idx),64,64,64);
            iso_thresh = 0.5.*(median(funGT(funGT>0),'all') + max(funGT,[],'all'));
            iso = isosurface(Z,X,Y,funGT,iso_thresh);
            p = patch(iso);
            set(p,'FaceColor',[0.3 0.3 0.3],'FaceAlpha',0.4);
            camlight;
            lighting gouraud;
            set(p,'EdgeColor','none');
            drawnow
            daspect([1 1 1])
        end
    
    end
end

%% Functions to be called by class methods, not to be called directly

function mesh = prepare_mesh(mesh)
    % Initialises geometric information such as finding boundary nodes etc.
    
    % Speed of light
    mesh.c = mesh.cVac./mesh.nr;
    
    % 3D Element Volume calculation
    aVec = mesh.node(mesh.elem(:,2),:)-mesh.node(mesh.elem(:,1),:);
    bVec = mesh.node(mesh.elem(:,3),:)-mesh.node(mesh.elem(:,1),:);
    cVec = mesh.node(mesh.elem(:,4),:)-mesh.node(mesh.elem(:,1),:);
    mesh.vol = abs(dot(cross(aVec,bVec,2),cVec,2)./6);
    
    % Delination and characterisation of 2D surface elements
    
    % EXTRACT SURFACE MESH
    % adapted from NIRFASTer boundfaces.m
    % Found here: http://www.nirfast.co.uk/downloads.html
    
    % faces of every element
    faces = [mesh.elem(:,[1,2,3]);...
           mesh.elem(:,[1,2,4]);...
           mesh.elem(:,[1,3,4]);...
           mesh.elem(:,[2,3,4])];
    % indexes for unique values
    [~,ix,jx]=unique(sort(faces,2),'rows');
    % indexes faces that only appear once i.e are outward facing
    vec = histc(jx,1:max(jx));
    qx = vec == 1;
    mesh.face=faces(ix(qx),:); % connection vectors for surface faces
    
    % TAGS SURFACE NODES WITH BINARY VALUES
    % 1 = boundary
    % 0 = non-boundary
    idx = unique(mesh.face(:));
    mesh.bnd = zeros(size(mesh.node,1),1);
    mesh.bnd(idx) = 1;
    
    % SURFACE ELEMENT CENTRE OF MASS
    x = (sum(reshape(mesh.node(mesh.face.',1),3,[]))./3).';
    y = (sum(reshape(mesh.node(mesh.face.',2),3,[]))./3).';
    z = (sum(reshape(mesh.node(mesh.face.',3),3,[]))./3).';
    com = [x y z];
    
    % ORIENT FACE ELEMENT NORMAL TO BE OUTWARD FACING AND CALCULATES AREA
    % viewing direction, defined as face CoM to mesh CoM
    centre = [mean([max(mesh.node(:,1)) min(mesh.node(:,1))]),...
            mean([max(mesh.node(:,2)) min(mesh.node(:,2))]),...
            mean([max(mesh.node(:,3)) min(mesh.node(:,3))])]; % mesh CoM
    vDir = (centre-com)./vecnorm(centre-com,2,2); % view vector
    % checks if normal is outward facing by dot product sign and corrects
    aVec = mesh.node(mesh.face(:,2),:)-mesh.node(mesh.face(:,1),:);
    bVec = mesh.node(mesh.face(:,3),:)-mesh.node(mesh.face(:,1),:);
    signA = sign(dot(cross(aVec,bVec,2),vDir,2));
    mesh.face(signA>0,[end-1,end]) = mesh.face(signA>0,[end,end-1]);
    aVec = mesh.node(mesh.face(:,2),:)-mesh.node(mesh.face(:,1),:);
    bVec = mesh.node(mesh.face(:,3),:)-mesh.node(mesh.face(:,1),:);
    normFace = cross(aVec,bVec,2);
    mesh.area = sqrt(sum(normFace.^2,2)).*0.5;
    
    % Boundary conditions and stiffness mapping
    
    % FACTOR FOR BOUNDARY CONDITIONS
    % assumes medium-air boundary
    R0 = ((mesh.nr-1).^2)./((mesh.nr+1).^2);
    thetaC = asin(1./mesh.nr);
    A = (2./(1-R0)-1+abs(cos(thetaC)).^3)./(1-abs(cos(thetaC)).^2);
    mesh.R = 1./(2.*A).*mesh.bnd;
    
    % MAPPING FOR STIFFNESS MATRIX
    % gradScale is used to map gradient from reference element to real element
    idx = mesh.elem.';
    idx = idx(:);
    temp = mesh.node(idx,:).';
    temp = reshape(temp,3,4,[]);
    B = temp(:,2:end,:) - temp(:,1,:);
    BT = pagetranspose(B);
    invBT = pagemldivide(BT,eye(size(BT,1)));
    invBT = reshape(invBT,9,[]);
    mesh.gradScale = invBT.';
    
    % Conversion to integers where needed
    mesh.elem = int32(mesh.elem);
    mesh.face = int32(mesh.face);
    mesh.bnd = int32(mesh.bnd);

end

% generates forward matrices, stores only upper diagonal
function [K,M] = gen_fwdmat(mesh)
    % Generates upper diagonal of forward matrix components (K,M)
    
    % INPUT
    % MESH = DOGPUP dMesh Object
    
    % OUTPUT
    % K = lower triangle of forward stiffness matrix, attenuation dependant component (n x n)
    % M = lower triangle of forward mass matrix, for frequency dependent component (n x n)
    
    % This version is based on theory from
    % Introduction to finite element methods, Hans Petter Langtangen
    % https://www.uio.no/studier/emner/matnat/ifi/IN5270/h20/ressurser/fem-book-4print.pdf
    % and
    % A gentle introduction to the Finite Element Method, Francisco–Javier Sayas
    % https://team-pancho.github.io/documents/anIntro2FEM_2015.pdf
    % super useful resources
    
    % Reference Element stuff
    % ref tet vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    % ref triangle vertices (0,0), (1,0), (0,1)
    % gauss-quadrature points and weights for reference tetrahedron
    [x3a,y3a,z3a,w3t]=TetQuadDat('GPU');
    % gauss-quadrature points and weights for reference triangle
    [x2a,y2a,w2t]=TriQuadDat('GPU');
    
    % basis function value in reference tetra at quadrature points
    N3D = [1-x3a-y3a-z3a; x3a; y3a; z3a;];
    N3Dgrad = gpuArray([-1,-1,-1; 1,0,0; 0,1,0; 0,0,1]);
    % basis functions in reference triangle at quadrature points
    N2D = [1-x2a-y2a; x2a; y2a;];
    
    clearvars x3a y3a z3a x2a y2a
    
    % VOLUME INTEGRATION - Mass and Stiffnes
    
    % volumes of elements distributed along array pages
    vol = reshape(mesh.vol,1,1,[]);
    
    % mapping to full mass+siffness matrix for each element matrix
    idx = mesh.elem.';
    idx = idx(:);
    elemMapI = reshape(idx,4,1,[]);
    elemMapI = repmat(elemMapI,1,size(mesh.elem,2),1);
    elemMapJ = pagetranspose(elemMapI);
    
    % nodal values for each element distrubuted along array pages
    kappa = gpuArray(reshape(mesh.kappa(idx),4,1,[]));
    mua = gpuArray(reshape(mesh.mua(idx),4,1,[]));
    % interpoalted to integration points for each element
    kappa = pagemtimes(N3D.',kappa);
    mua = pagemtimes(N3D.',mua);
    % weighted for integration
    kappa = pagefun(@times,w3t.',kappa);
    mua = pagefun(@times,w3t.',mua);
    
    % stiffness transform to each element from reference
    B = gpuArray(reshape(mesh.gradScale.',3,3,[]));
    
    % element matrix for each property
    elemMatK = gpuArray(zeros(size(mesh.elem,2),size(mesh.elem,2),size(mesh.elem,1)));
    elemMatM = elemMatK;
    
    [iloc,jloc] = ind2sub([size(mesh.elem,2),size(mesh.elem,2)],1:size(mesh.elem,2)^2);
    
    % generate element matrices for K and M
    for n = 1:length(iloc)
    
        basisI = N3D(iloc(n),:).';
        basisJ = N3D(jloc(n),:).';
        gradI = pagemtimes(B,N3Dgrad(iloc(n),:).');
        gradJ = pagemtimes(B,N3Dgrad(jloc(n),:).');
    
        % stiffness matrix
        preFact = gradI.*gradJ;
        preFact = sum(preFact,1);
        preFact = pagefun(@times,vol,preFact);
        stiff = pagefun(@times,preFact,sum(kappa,1));
    
        % absorption mass matrix
        absrp = pagefun(@times,vol,sum(mua.*basisI.*basisJ,1));
    
        elemMatK(iloc(n),jloc(n),:) = stiff + absrp;
    
        % mass matrix
        mass = pagefun(@times,vol,sum(w3t.'.*basisI.*basisJ,1));
        elemMatM(iloc(n),jloc(n),:) = mass;
    
    end
    
    K = sparse(elemMapI(:),elemMapJ(:),gather(elemMatK(:)),size(mesh.node,1),size(mesh.node,1));
    M = sparse(elemMapI(:),elemMapJ(:),gather(elemMatM(:)),size(mesh.node,1),size(mesh.node,1));
    
    % SURAFCE INTEGRATION

    % first find boundary face elements
    tempFace = sort(mesh.elem,2);
    tempFace = gpuArray(tempFace.*mesh.bnd(tempFace));
    % process tetrahedrons with all boundary nodes by splitting into component
    % faces
    mask = sum(sign(tempFace),2);
    bFace = tempFace(mask==4,:);
    bFace = cat(1,bFace(:,[1,2,3]),bFace(:,[1,2,4]),...
        bFace(:,[1,3,4]),bFace(:,[2,3,4]));
    % add remaining faces
    tempFace = tempFace(mask==3,:);
    tempFace = reshape(nonzeros(tempFace.'),3,[]).';
    bFace = cat(1,bFace,tempFace);
    
    % find area of each face
    aVec = mesh.node(bFace(:,2),:)-mesh.node(bFace(:,1),:);
    bVec = mesh.node(bFace(:,3),:)-mesh.node(bFace(:,1),:);
    normFace = cross(aVec,bVec,2);
    area = sqrt(sum(normFace.^2,2)).*0.5;
    
    % areas of face elements distributed along array pages
    area = reshape(area,1,1,[]);
    
    % mapping to full matrix for each element matrix
    idx = bFace.';
    idx = idx(:);
    elemMapI = reshape(idx,3,1,[]);
    elemMapI = repmat(elemMapI,1,size(bFace,2),1);
    elemMapJ = pagetranspose(elemMapI);
    
    % nodal values for each element distrubuted along array pages
    R = gpuArray(reshape(mesh.R(idx),3,1,[]));
    % interpoalted to integration points for each element
    R = pagemtimes(N2D.',R);
    % weighted for integration
    R = pagefun(@times,w2t.',R);
    
    elemMatK = gpuArray(zeros(size(bFace,2),size(bFace,2),size(bFace,1)));
    
    [iloc,jloc] = ind2sub([size(bFace,2),size(bFace,2)],1:size(bFace,2)^2);
    
    % generate element matrices for boundary integration
    for n = 1:length(iloc)
    
        basisI = N2D(iloc(n),:).';
        basisJ = N2D(jloc(n),:).';
    
        % boundary condition matrix
        bound = pagefun(@times,area,sum(R.*basisI.*basisJ,1));
        elemMatK(iloc(n),jloc(n),:) = bound;
    
    end
    
    K = K + sparse(elemMapI(:),elemMapJ(:),gather(elemMatK(:)),size(mesh.node,1),size(mesh.node,1));
    
    % clear GPU
    clearvars -except K M
    K = gather(K);
    M = gather(M);
    reset(gpuDevice);
    % assume that matrices are symmetric so we can just store upper diagonal
    K = triu(K);
    M = triu(M);
    
    

end

% Tetrahedral quadrature
function [xa,ya,za,wt]=TetQuadDat(type)
% Quadrature data for tetrahedron
% Refs
%  P Keast, Moderate degree tetrahedral quadrature formulas, CMAME 55: 339-348 (1986)
%  O. C. Zienkiewicz, The Finite Element Method,  Sixth Edition,
% From https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html

    xa= [1/4, 1/2, 1/6, 1/6, 1/6];
    ya= [1/4, 1/6, 1/6, 1/6,  1/2];
    za= [1/4, 1/6, 1/6,  1/2, 1/6];
    wt= [-0.8, 0.45, 0.45, 0.45, 0.45];
    
    if nargin > 0 && strcmp(type,'GPU') == 1
        xa = gpuArray(xa);
        ya = gpuArray(ya);
        za = gpuArray(za);
        wt = gpuArray(wt);
    end

end

% Triangular quadrature
function [xa,ya,wt] = TriQuadDat(type)
    % https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tri/quadrature_rules_tri.html
    %*****************************************************************************80
    %
    %% Triangle quadrature for order 2
    
    xa = [2/3, 1/6, 1/6];
    ya = [1/6, 2/3, 1/6];
    wt = [1/3 1/3 1/3];
    
    if nargin > 0 && strcmp(type,'GPU') == 1
    xa = gpuArray(xa);
    ya = gpuArray(ya);
    wt = gpuArray(wt);
end

end

% Solve fluence for given source
function phi = fluGPU(mesh,Q)
    % generate full forward matrices and move matrices to GPU
    Q = gpuArray(Q);
    A = mesh.K + 1j.*mesh.optode.df./mesh.c.*mesh.M;
    A = gpuArray(A + triu(A,1).'); % full fwd matrix
    phi = gather(gpuBicstab_FSAIP(mesh,A,Q,1e-12,1e3)); % solves for phiA = nodes x freq x source
end

% BiCGStab with FSAI precon to solve diffusion approx.
function x = gpuBicstab_FSAIP(mesh,A,Q,tol,iter)
%GPUBICSTAB uses GPU parallelised bicgstab algo to solve linear system
% assumes Q input is nodes x freqs x sources

%% Initialisation

% generate FSAI preconditioner
[cPtrG,rPtrG,G] = FSAIP_gen(mesh,A);
rGfull = int32(repelem((1:(size(rPtrG,1)-1)).',diff(rPtrG)));
[cPtrGT,rPtrGT,GT] = sparse_csr(cPtrG+1,rGfull,G);

Q = gpuArray(permute(Q,[1 3 2])); % nodes x sources x freqs
freqs = size(Q,3);
f = gpuArray(0:freqs-1); % Fourier series frequency integers

% represent sparse matrices in CSR format
[cPtrA,rPtrA,A] = sparse_csr(A);

% move to GPU
cPtrG = gpuArray(cPtrG);
rPtrG = gpuArray(rPtrG);
G = gpuArray(G);
cPtrGT = gpuArray(cPtrGT);
rPtrGT = gpuArray(rPtrGT);
GT = gpuArray(GT);
cPtrA = gpuArray(cPtrA);
rPtrA = gpuArray(rPtrA);
A = gpuArray(A);
tol = gpuArray(tol);

clearvars rGfull

%% Bicgstab

% bicgstab initialisation
% compute intial guess
x = gpuArray(complex(zeros(size(Q))));
r = smv_mex(x,f,A,rPtrA,cPtrA);
r = Q - r;
err0 = sqrt((sum(abs(r).^2,1)));

rhat0 = r;
rho0 = sum(rhat0.*r,1);
p = r;

% main loop
for i = 1:iter

    h = FSAImv_mex(p,G,rPtrG,cPtrG);
    h = FSAImv_mex(h,GT,rPtrGT,cPtrGT);
    wait(gpuDevice);

    

    v = smv_mex(h,f,A,rPtrA,cPtrA);
    wait(gpuDevice);

    alph = sum(v.*rhat0,1)./rho0;
    h = x + h./alph;
    s = r - v./alph;
    err = sqrt(sum(abs(s).^2,1))./err0;
    err = max(err,[],'all');

    if err < tol
        x = h;
        break  
    end
    
    Gs = FSAImv_mex(s,G,rPtrG,cPtrG);
    z = FSAImv_mex(Gs,GT,rPtrGT,cPtrGT);
    wait(gpuDevice);

    t = smv_mex(z,f,A,rPtrA,cPtrA);
    wait(gpuDevice);

    Gtt = FSAImv_mex(t,G,rPtrG,cPtrG);
    wait(gpuDevice);

    om = sum(Gtt.*Gs,1)./sum(Gtt.*Gtt,1);

    x = h + om.*z;
    r = s - om.*t;
    p = p - om.*v;
    err = sqrt((sum(abs(r).^2,1)))./err0;
    err = max(err,[],'all');

    if err < tol
        break  
    end

    rho = sum(rhat0.*r,1);
    beta = rho./rho0.*(1./(alph.*om));
    rho0 = rho;
    p = r + beta.*p;
    
end

clearvars -except x
x = gather(permute(x,[1 3 2])); % nodes x freqs x sources
reset(gpuDevice);

end
    
% Generates FSAI precon for solving system of equations
function [cPtrG,rPtrG,valG] = FSAIP_gen(mesh,A)
    %FSAIP_GEN Generates FSAIP preconditioners for forward matrix in CSR format
    % at given frequency integer
    % see:
    % https://pmc.ncbi.nlm.nih.gov/articles/PMC5709934/
    % https://onlinelibrary.wiley.com/doi/10.5402/2012/127647
    
    %% Find Sparsity Pattern of preconditioner
    
    % fwd matrix lower diagonal
    A = gather(tril(A));
    
    N = 30;
    
    [i,j,v] = find(A);
    % find 3 max abs vals in each row
    spPattern = cat(2,i,abs(v));
    [spPattern,idx] = sortrows(spPattern,'descend');
    idxMax = diff([spPattern(end,1)+1; spPattern(:,1)]);
    idxMax = find(idxMax);
    idxMax = unique(repmat(idxMax,N,1) - kron((0:N-1).',ones(size(idxMax))));
    idxMax = idxMax(idxMax>0);
    idxMax = idx(idxMax); % index of 3 largest abs val in each row
    spPattern = [i(idxMax),j(idxMax)];
    spPattern = sortrows(spPattern);
    
    %% Convert Sparsity and Forward matrix to CSR
    
    [cPtrG,rPtrG,~] = sparse_csr(spPattern(:,1),spPattern(:,2),ones(size(spPattern,1),1));
    [cPtrA,rPtrA,valA] = sparse_csr(A);
    
    %% Find FSAI for each frequency
    
    valG = FSAIP_gen_mex(valA,rPtrA,cPtrA,rPtrG,cPtrG,int32(length(mesh.optode.fAxis))).';
end

% Converts MATLAB sparse to CSR
function [c,r,val] = sparse_csr(varargin)

if length(varargin) == 1
    A = varargin{1};
    [c,r,val] = find(A.');
    val = val;
elseif length(varargin) == 3
    r = varargin{1};
    c = varargin{2};
    val = varargin{3};
    [r,idx] = sort(r);
    c = c(idx);
    sz = size(val);
    if sz(1) == length(idx)
        val = val(idx,:);
    else
        val = val(:,idx);
    end
else
    error('Input must be MATLAB sparse array or row and col and val of sparse non-zeros');
end

c = int32((c-1));
r = accumarray(r+1,1);
r = int32((cumsum(r)));

end

% toggle display of text
function tdisp(textIn,toggle)
    if nargin == 2 && logical(toggle) == 1
        fprintf(textIn)
    elseif ~islogical(toggle)
        error('displayFlag must be logical')
    end
end

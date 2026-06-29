classdef dMesh < matlab.mixin.Copyable
    % DOGPUP mesh class

    properties (Access = public)
        % MESH GEOMETRY
        node % mesh nodes
        elem % mesh tetrehedral elements
        vol % mesh tetra volumes
        face % mesh face triangle elements
        area % mesh triangle areas
        bnd % binary boundary marker

        % MESH PROPERTIES
        nr % global refractive index
        c % speed of light in media mms^-1
        mua % absorption coeff mm^-1
        musp % reduced scattering coeff mm^-1
        kappa % diffusion coeff mm
        R % boundary factor

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
        cVac = 2.99792458e11; % speed of light in vaccum mms^-1
    end
    
    % Proper copy method for optode handle
    methods(Access = protected)
      function cpy = copyElement(obj)
         cpy = copyElement@matlab.mixin.Copyable(obj);
         cpy.optode = copy(obj.optode);
      end
    end
    
    methods 
        %% Mesh Setup/Utils
        % Construct an instance of this class
        function mesh = dMesh(varargin)
            % Construct DOGPUP mesh from given mesh geometry and optical
            % properties

            % INPUT
            % varargin{1} = nodes [NN x 3] or path to mesh json
            % varargin{2} = mesh connectivity list [NE x 4]
            % varargin{3} = refractive index of mesh [scalar]
            % varargin{4} = optical properties of mesh [mua musp], either [1 x 2] or [NN x 2]

            % mua = absoprtion (mm^-1), musp = reduced scattering (mm^-1)
            % NN = number of nodes, NE = number of elements, 

            % OUTPUT
            % mesh = DOGPUP mesh class

            if nargin == 1
                mesh = load_mesh(varargin{1});
            else
                mesh.node = varargin{1};
                mesh.elem = varargin{2};
                mesh.nr = varargin{3};
                mesh = prepare_mesh(mesh); % prepare geometry
                mesh.update_properties(varargin{4}) % fill optical props
            end
        end

        % Update optical properties
        function update_properties(mesh,props)
            % Asigns optical properties at each node of the mesh

                if isempty(props)
                    mesh.mua = [];
                    mesh.musp = [];
                    mesh.kappa = [];
                else
                    mesh.mua = zeros([size(mesh.node,1),1]);
                    mesh.musp = zeros([size(mesh.node,1),1]);
                    mesh.kappa = zeros([size(mesh.node,1),1]);
                    mesh.mua(:) = props(:,1);
                    mesh.musp(:) = props(:,2);
                    mesh.kappa(:) = 1./(3.*(mesh.mua + mesh.musp));
                end
        end

        % link optode class to mesh class
        function add_optode(mesh,optode)
            % snaps optodes to nearest surface of the mesh and links the
            % two objects
            optode.snap2mesh(mesh);
            mesh.optode = optode;
        end
        
        % clear optodes
        function mesh = clear_optode(mesh)
            % unlinks optodes
            mesh.optode = [];
        end
    
        % save mesh as json
        function save_mesh(mesh,fn,overwrite_flag)
            % save DOGPUP mesh as json
            
            % mesh = DOGPUP mesh object
            % fn = string or character filename to be written

            if ~isletter(fn)
                error('fn must be a string or character')
            end

            % check if file exists
            fn = string(fn);
            [fpath,fn,~] = fileparts(fn);
            if nargin > 2 && strcmp(string(overwrite_flag),'overwrite')
                fn = fn;
            elseif nargin > 2 && ~isletter(overwrite_flag)
                error('overwrite_flag must be string or char')
            elseif nargin < 3 || ~strcmp(string(overwrite_flag),'overwrite')
                if nargin > 2 && ~strcmp(string(overwrite_flag),'overwrite')
                    warning('Invalid overwrite_flag must be string ''overwrite'', writing to new file')
                end
                listing = dir(fpath + "\" + fn + "*.json");
                if ~isempty(listing)
                    fn = fn + "(" + num2str(length(listing)) + ")";
                end
            end
            fn = fpath + "\" + fn + ".json";

            % mesh geometry
            structOut.geometry.nodes = mesh.node;
            structOut.geometry.elements = mesh.elem;
            structOut.parameters.refractiveIndex = mesh.nr;
            structOut.parameters.opticalProps = [mesh.mua mesh.musp];
            % mesh optode
            structOut.optode.source.position = [mesh.optode.s_positions mesh.optode.s_bary];
            structOut.optode.source.direction = mesh.optode.s_dirs;
            structOut.optode.detector.position = [mesh.optode.d_positions mesh.optode.d_bary];
            structOut.optode.detector.direction = mesh.optode.d_dirs;
            structOut.optode.link = mesh.optode.link;
            if size(mesh.optode.ch_tirf,1) == length(mesh.optode.tAxis(:))
                structOut.optode.timeData = [mesh.optode.tAxis(:) mesh.optode.ch_tirf];
            else
                structOut.optode.timeData = [mesh.optode.tAxis(:) mesh.optode.ch_tirf.'];
            end
            if size(mesh.optode.ch_firf,1) == length(mesh.optode.fAxis(:))
                structOut.optode.freqData = [mesh.optode.fAxis(:) mesh.optode.ch_firf];
            else
                structOut.optode.freqData = [mesh.optode.fAxis(:) mesh.optode.ch_firf.'];
            end

            json = savejson('DOGPUP_mesh',structOut);
            fid = fopen(fn,'w');
            fprintf(fid, '%s', json);
            fclose(fid);
        end

        
        %% Fluence solving methods
        % solves for forward fluence using BICGSTAB and FSAI preconditioning
        function [phi,data] = get_fluence(mesh,displayFlag)
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
                displayFlag = true;
            end

            % generate source vectors
            Q = zeros(size(mesh.node,1),1,size(mesh.optode.s_positions,1));
            for i = 1:size(mesh.optode.s_positions,1)
                Q(mesh.elem(mesh.optode.s_bary(i,1),:),:,i) = mesh.optode.s_bary(i,2:end);
            end
            Q = repmat(Q,1,mesh.optode.Nf,1);
            % solve
            tdisp('\nsolving for fluence...',displayFlag)
            phi = fluGPU(mesh,Q);
            tdisp('done!\n',displayFlag)

            % get detector data
            if nargout > 1
                data = mesh.get_detections(phi);
                tdisp('detection generated!\n',displayFlag)
            end

        end

        % gets data from detectors
        function data = get_detections(mesh,phi)
            % Finds fluence at detectors
            
            % INPUT
            % mesh = fully initialised DOGPUP mesh
            % phi = Fourier coefficients of fluence through mesh (NN x NF x NS)

            % OUTPUT
            % data = Fourier coefficients of fluence at detectors (NM x NF)

            i = kron((1:size(mesh.optode.d_positions,1)).',ones(4,1));
            i = i(mesh.optode.d_bary(:,2:end).'>0);
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
            data = data.*mesh.optode.ch_firf;
        end
        
        % solves for adjoint fluence using BICGSTAB and FSAI preconditioning
        function phiA = get_adjoint(mesh,displayFlag)
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
            Q = repmat(Q,1,mesh.optode.Nf,1);
            % solve
            tdisp('\nsolving for adjoint fluence...',displayFlag)
            phiA = fluGPU(mesh,Q);
            tdisp('done!\n',displayFlag)

        end

        % generates complex FD jacobian
        function [J,data,phi,phiA] = J_complex(mesh,phi,phiA,type,displayFlag)
            % Finds the Fourier series coefficient absorption Jacobian 
            % using the adjoint method

            % INPUT
            % mesh = fully initialised DOGPUP mesh
            % displayFlag = flag for text display (boolean)
            % type = 'full' or empty, full calculates both absorption and kappa
            
            % OUTPUT
            % J = sensitivity for Fourier coefficients (NM x NF x NV or NM x NF x 2*NV)
            % data = Fourier coefficients of fluence at detectors (NM x NF)
            % phi = Fourier coefficients of fluence through mesh (NN x NF x NS)
            % phiA = Fourier coefficients of adjoint fluence through mesh (NN x NF x ND)

            % NN = number of nodes, NF = number of frequencies, 
            % NV = number of voxels, NS = number of sources, 
            % ND = number of detectors, NM = number of measurements

            if nargin < 5
                displayFlag = 1;
            end

            if nargin < 4 || isempty(type)
                type = 'mua';
            end
            
            if nargin < 3 || isempty(phiA)
                % generate adjoint fluence if not given
                phiA = get_adjoint(mesh,displayFlag);
                if nargin < 2 || isempty(phi)
                    % generate forward fluence if not given
                    phi = get_fluence(mesh,displayFlag);
                end
            end

            data = get_detections(mesh,phi);
            
            tdisp('\ngenerating complex absorption jacobian...',displayFlag)
            % interpolate fluences to grid
            phi_Grid = mesh.m2g*reshape(phi(:,:,mesh.optode.link(:,1)),size(phi,1),[]);
            phiA_Grid = mesh.m2g*reshape(phiA(:,:,mesh.optode.link(:,2)),size(phiA,1),[]);
            % correctly reshape to voxels
            J = -phi_Grid.*phiA_Grid.*mesh.dxyz^3; % voxel_mua x freq*meas
            J = reshape(J,size(phi_Grid,1),[],size(mesh.optode.link,1)); % voxel_mua x freq x meas
            J = permute(J,[3 2 1]); % meas x freq x voxel_mua
            
            tdisp('done!\n',displayFlag)

            if strcmp(type,'full')

                tdisp('\ngenerating complex reduced scattering jacobian...',displayFlag)
                % compute gradient with matrices
                [Dx,Dy,Dz] = grid_grad(mesh);
                % dot product of gradients
                Jk = (Dx*phi_Grid).*(Dx*phiA_Grid) + (Dy*phi_Grid).*(Dy*phiA_Grid) + (Dz*phi_Grid).*(Dz*phiA_Grid);
                Jk = Jk.*mesh.dxyz^3; % voxel_kappa x freq*meas
                Jk = reshape(Jk,size(phi_Grid,1),[],size(mesh.optode.link,1)); % voxel_kappa x freq x meas
                Jk = Jk.*(mesh.m2g*(1./(3.*mesh.kappa.^2)));
                Jk = permute(Jk,[3 2 1]); % meas x freq x voxel_kappa

                J = cat(3,J,Jk); % meas x freq x [voxel_mua voxel_kappa]
                tdisp('done!\n',displayFlag)
            end
            J = J.*mesh.optode.ch_firf; % convolve with channel IRF
        end


        %% Slicing and interpolation methods
        % Grid interpolation for reconstruction basis
        function mesh2grid(mesh,x,y,z)
        % Generates transformation matrices for interpolation to voxel
        % grid and vice versa

        % INPUT
        % mesh = DOGPUP mesh
        % [x,y,z] = grid points [NV x 3] (mm)

        % OUTPUT
        % mesh.grid = grid points [NV x 3] (mm)
        % mesh.gridSize = number of voxels in x,y and z [1 x 3]
        % mesh.dxyz = resolution of voxel grid
        % mesh.m2g = interpolation matrix from mesh to grid [NV (in mesh) x NN]
        % mesh.g2m = interpolation matrix from grid to mesh [NN x NV (in mesh)]
        % mesh.gridinmesh = binary flag, true if voxel is inside mesh [NV x 1]

        % NV = number of voxels, NN = number of nodes

        % reset grid
        mesh.reset_grid;
        
        % assumes cubic grid

        % define grid
        dx = abs(x(2) - x(1));
        [X,Y,Z] = ndgrid(x,y,z);
        mesh.grid = [X(:),Y(:),Z(:)];
        mesh.gridSize = [length(x), length(y), length(z)];
        mesh.dxyz = dx;
        
        %% Calculate mesh to grid transform matrix
        % find barycentric co-ords of each voxelised point w.r.t mesh
        TR = triangulation(double(mesh.elem),mesh.node);
        [idInt,bary] = pointLocation(TR,mesh.grid);
        mesh.gridinmesh = logical(~isnan(idInt));
        
        % use barycentric co-ords to form interpolation matrix
        i = (1:sum(mesh.gridinmesh)).';
        i = repelem(i,4,1);
        
        j = double(mesh.elem(idInt(mesh.gridinmesh),:)).';
        j = j(:);
        
        v = bary(mesh.gridinmesh,:).';
        v = v(:);
        
        % construct mesh to grid interpolation matrix
        mesh.m2g = sparse(i,j,v,sum(mesh.gridinmesh),size(mesh.node,1));
        
        %% Calculate grid to mesh transform matrix
        % find barycentric co-ords of each node w.r.t voxel grid
        % create reference triangulation of 8 voxels
        origin = mesh.grid(1,:);
        [X,Y,Z] = ndgrid([0 dx]);
        node_vox0 = [X(:),Y(:),Z(:)];
        elem_vox0 = delaunay(node_vox0);
        % rescale mesh nodes such that their co-ords are relative to this
        % reference
        origin_rel = floor((mesh.node - origin) / dx);
        node_rel = mesh.node - (origin_rel.*dx + origin);
        node_rel(abs(node_rel)<1e-14) = 0;
        % find barycentric co-ords in the reference voxel
        TR = triangulation(elem_vox0,node_vox0);
        [ind,bary] = pointLocation(TR,node_rel);
        % convert to real voxel

        % find global index of each nodes voxel origin
        origin_idx = origin_rel + 1;
        % handle nodes outside grid span
        idx = any(origin_idx < 1,2) | any(origin_idx > mesh.gridSize,2);
        if sum(idx) > 0
            origin_idx(idx,:) = ones(sum(idx),3);
        end
        origin_idx = sub2ind(mesh.gridSize,origin_idx(:,1),origin_idx(:,2),origin_idx(:,3));

        % offset voxel origin idx
        rel_idx = node_vox0(:,1)./dx + mesh.gridSize(:,1).*(node_vox0(:,2)./dx) + prod(mesh.gridSize(:,1:2)).*(node_vox0(:,3)./dx);
        true_idx = origin_idx + rel_idx(elem_vox0(ind,:));
        true_idx(idx,:) = ones(sum(idx),4).*find(~mesh.gridinmesh,1);
        true_idx(true_idx > numel(mesh.gridinmesh)) = find(~mesh.gridinmesh,1);
        mask = mesh.gridinmesh(true_idx);
        % tag mesh nodes with voxel vertices in grid (where grid is now defined as voxels in orignal mesh)
        meshingrid = any(mask>0,2);
        bary = mask.*bary;
        
        % use barycentric co-ords to form interpolation matrix
        % removes weighting from voxel points outside mesh

        i = 1:size(mesh.node,1);
        i = i(meshingrid).';
        i = repelem(i,4,1);
        
        j = true_idx(meshingrid,:).';
        j = j(:);
        LUT_j = cumsum(mesh.gridinmesh);
        j = LUT_j(j);
        
        bary = bary(meshingrid,:).';
        v = bary(:);

        i = i(j>0);
        v = v(j>0);
        j = j(j>0);
        
        % find nodes that are outside voxel grid
        if sum(meshingrid) < size(mesh.node,1)
        
            % nearest neighbour interpolation for nodes that lie in the zero space
            idx = ~meshingrid;
            iS = 1:size(mesh.node,1);
            iS = iS(idx).';
            jS = dsearchn(mesh.grid(mesh.gridinmesh,:),mesh.node(idx,:));
            vS = ones(length(iS),1);
            
            i = [i; iS];
            j = [j; jS];
            v = [v; vS];
        
        end
        
        % construct grid to mesh interpolation matrix
        mesh.g2m = sparse(i,j,v,size(mesh.node,1),sum(mesh.gridinmesh));
        norm = sum(mesh.g2m,2);
        norm(norm==0) = 1;
        mesh.g2m = mesh.g2m./norm;
        
        
        end

        % reset grid properties
        function reset_grid(mesh)
        % resets grid co-ords and interpolation matrices
            mesh.grid = [];
            mesh.gridSize = [];
            mesh.dxyz = [];
            mesh.m2g = [];
            mesh.g2m = [];
            mesh.gridinmesh = [];

        end

        % generatie derivative matrices
        function [Dx,Dy,Dz] = grid_grad(mesh)
        % Generate derivative matrices for grid
            % sizes
            N_x = mesh.gridSize(1);
            N_y = mesh.gridSize(2);
            N_z = mesh.gridSize(3);
            N = N_x*N_y*N_z;

            % identity matrices for Kronecker products
            Ix = speye(N_x);
            Iy = speye(N_y);
            Iz = speye(N_z);

            % mask matrix
            W = spdiags(double(mesh.gridinmesh),0,N,N);
  
            % mask voxel image
            mask = reshape(mesh.gridinmesh, N_x, N_y, N_z);
            
            for d = 1:3
                % compute central difference matrix for dimension d
                switch d
                    case 1 % x derivative
                        D = spdiags([-1/2 1/2]./mesh.dxyz,[-1 1],N_x,N_x);
                        D = kron(Iz, kron(Iy, D));
                        v_idx = 1;
                    case 2 % y derivative
                        D = spdiags([-1/2 1/2]./mesh.dxyz,[-1 1],N_y,N_y);
                        D = kron(Iz, kron(D, Ix));
                        v_idx = N_x;
                    case 3 % z derivative
                        D = spdiags([-1/2 1/2]./mesh.dxyz,[-1 1],N_z,N_z);
                        D = kron(D, kron(Iy, Ix));
                        v_idx = N_x*N_y;
                end

                % mask external voxels
                D = W*D*W;

                % boundary index
                boundary = (mask & (~circshift(mask,1,d) | ~circshift(mask,-1,d)));
                boundary = boundary(:);
                % forward/backward difference at boundaries
                D(boundary,:) = 0;
                i = find(boundary);
                j = [i-v_idx i i+v_idx];
                j(~ismember(j,find(mesh.gridinmesh))) = 0;
                j = sort(j,2);
                j = j(:,2:3);

                % find isolated voxels
                idx = sum(logical(j),2) < 2;
                i_s = i(idx);
                
                % mask and boundary derivative
                i = i(~idx);
                i = repelem(i,1,2);
                j = j(~idx,:);
                v = repelem([-1 1],size(j,1),1)./mesh.dxyz;
                D = D + sparse(i(:),j(:),v(:),size(D,1),size(D,1));
                
                % interploation matrix from nearest values for
                % isolated voxels
                j_s = [i_s-N_x*N_y i_s-N_x i_s-1 i_s+1 i_s+N_x i_s+N_x*N_y];
                j_s(~ismember(j_s,find(mesh.gridinmesh))) = 0;
                i_s = repelem(i_s,1,6);
                v_s = 1./sum(logical(j_s),2);
                v_s(isinf(v_s)) = 0;
                v_s = repelem(v_s,1,6);

                I_mat = sparse(i_s(j_s>0),j_s(j_s>0),v_s(j_s>0),N,N) + speye(N);
                D = I_mat*D;
                
                switch d
                    case 1 % x derivative
                        Dx = D;
                    case 2 % y derivative
                        Dy = D;
                    case 3 % z derivative
                        Dz = D;
                end

            end

            % restrict to voxels inside mesh
            Dx = Dx(mesh.gridinmesh,mesh.gridinmesh);
            Dy = Dy(mesh.gridinmesh,mesh.gridinmesh);
            Dz = Dz(mesh.gridinmesh,mesh.gridinmesh);

        end

        % function to interpolate values to different mesh
        function val_int = mesh2mesh(mesh,val,old_node,old_elem)
            % interplotes values between two meshes

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
                val_int(~isnan(ind),i) = sum(bary(~isnan(ind),:).*temp(old_elem(ind(~isnan(ind)),:)),2);
            end

            % fill outlying nodes with nearest neighbours
            out_node = mesh.node(isnan(ind),:);
            in_node = mesh.node(~isnan(ind),:);
            val_in = val_int(~isnan(ind),:);
            k = dsearchn(in_node,out_node);
            val_int(isnan(ind),:) = val_in(k,:);

        end
        
        % function to slice mesh to grid for display forms 512 x 512 image
        function [sliceMat,points,plane] = mesh_slice(mesh,plane)
            % generates matrix transform to find slice of function defined
            % on mesh

            % INPUT
            % mesh = DOGPUP mesh
            % plane = string that is formated 'x=plane', 'y=plane' or
            % 'z=plane', determines slicing plance

            % OUTPUT
            % sliceMat = matrix to interpolate at slice [512*512 x NN]
            % points = (x,y,z) points on slice [512*512 x 3]

            % NN = number of nodes
            
            % check if string/char and remove whitespace
            if ischar(plane) == 0 && isstring(plane) == 0
                error('Char/Str input required for slicing plane')
            else
                plane = convertStringsToChars(plane);
                plane = erase(plane,' ');
            end

            if ~strcmpi(plane(1:2),'x=') && ~strcmpi(plane(1:2),'y=') && ~strcmpi(plane(1:2),'z=')
                error('Format for slicing plane is similar to ''x=30''')
            end

            res = 512;
            dx = max(abs(max(mesh.node) - min(mesh.node)))./(res-2);
            x = linspace(-dx*(res/2),dx*(res/2),res);
            
            if strcmp(plane(1:2),'x=')
                [X,Y,Z] = meshgrid(str2double(extractAfter(plane,'=')),mean(mesh.node(:,2)) + x, mean(mesh.node(:,3)) + x);
            elseif strcmpi(plane(1:2),'y=')
                [X,Y,Z] = meshgrid(mean(mesh.node(:,1)) + x,str2double(extractAfter(plane,'=')),mean(mesh.node(:,3)) + x);
            elseif strcmpi(plane(1:2),'z=')
                [X,Y,Z] = meshgrid(mean(mesh.node(:,1)) + x,mean(mesh.node(:,2)) + x,str2double(extractAfter(plane,'=')));
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

        % function to interpolate mesh to 3D grid for display forms 128 x 128 x 128
        % volume
        function [volMat,points] = mesh_slice3(mesh)
            % generates matrix transform to convert data to 128 x 128 x 128
            % grid for display purposes

            % INPUT
            % mesh = DOGPUP mesh

            % OUTPUT
            % volMat = matrix to interpolate to grid [128*128*128 x NN]
            % points = (x,y,z) points in volume [128*128*128 x 3]

            % NN = number of nodes
            
            res = 128;
            dx = max(abs(max(mesh.node) - min(mesh.node)))./(res-2);
            x = linspace(-dx*(res/2),dx*(res/2),res);
            [X,Y,Z] = meshgrid(mean(mesh.node(:,1)) + x,mean(mesh.node(:,2)) + x, mean(mesh.node(:,3)) + x);
            
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
            p0 = trimesh(mesh.face,mesh.node(:,3),mesh.node(:,1),mesh.node(:,2),'EdgeColor',[0.65 0.65 0.65],'FaceColor',[0.8 0.8 0.8],'FaceAlpha',alpha,'EdgeAlpha',alpha/3);
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
            set(gca,'XDir','normal','YDir','normal','ZDir','normal')
            view(-45,45)
            axis equal
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

            if  nargin == 1 || isempty(alpha)
                alpha = 1;
            end
            
            if alpha > 0
                plotdmesh(mesh,alpha)
            end
            hold on
            p1 = scatter3(mesh.optode.s_positions(:,3),mesh.optode.s_positions(:,1),mesh.optode.s_positions(:,2),20,'r','filled');
            p2 = scatter3(mesh.optode.d_positions(:,3),mesh.optode.d_positions(:,1),mesh.optode.d_positions(:,2),20,'b','filled');

            if nargin < 3 || lbl_flag == true
                % number sources and detectors
                stxt = strsplit(num2str(1:size(mesh.optode.s_positions,1))); % source labels
                dr = -3.*mesh.optode.s_dirs;
                txtPos = mesh.optode.s_positions;
                if ~isempty(dr)
                    txtPos = txtPos + dr;
                else
                    txtPos = txtPos + [1 1 1];
                end
                text(txtPos(:,3),txtPos(:,1),txtPos(:,2),stxt,'Color','red')
    
                dtxt = strsplit(num2str(1:size(mesh.optode.d_positions,1)));
                dr = -3.*mesh.optode.d_dirs;
                txtPos = mesh.optode.d_positions;
                if ~isempty(dr)
                    txtPos = txtPos + dr;
                else
                    txtPos = txtPos + [1 1 1];
                end
                text(txtPos(:,3),txtPos(:,1),txtPos(:,2),dtxt,'Color','blue')
            end

            % Format datatip
            mytip = datatip(p1);
            txt1 = dataTipTextRow('X','YData');
            txt2 = dataTipTextRow('Y','ZData');
            txt3 = dataTipTextRow('Z','XData');
            p1.DataTipTemplate.DataTipRows(1) = txt1;
            p1.DataTipTemplate.DataTipRows(2) = txt2;
            p1.DataTipTemplate.DataTipRows(3) = txt3;
            delete(mytip);

            % Format datatip
            mytip = datatip(p2);
            txt1 = dataTipTextRow('X','YData');
            txt2 = dataTipTextRow('Y','ZData');
            txt3 = dataTipTextRow('Z','XData');
            p2.DataTipTemplate.DataTipRows(1) = txt1;
            p2.DataTipTemplate.DataTipRows(2) = txt2;
            p2.DataTipTemplate.DataTipRows(3) = txt3;
            delete(mytip);

            % view, labels and scale
            xlabel('z (mm)')
            ylabel('x (mm)')
            zlabel('y (mm)')
            set(gca,'XDir','normal','YDir','normal','ZDir','normal')
            view(-45,45)
            daspect([1 1 1])

        end
    
        % plot slice through mesh
        function plotfun_slice(mesh,fun,map,plane,incl)
            % Plots 2D slice of function defined on mesh or grid
            
            % INPUT
            % mesh = DOGPUP mesh
            % fun = function defined on mesh [NN/NG x 1]
            % map = matlab colormap
            % plane = string that is formated 'x=plane', 'y=plane' or
            % 'z=plane', determines slicing plance
            % incl = inclusion array [NI X (x,y,z) radius]

            % NN = number of nodes, NG = number of grid points ,NI = number of inclusions

            if length(fun) == size(mesh.node,1)
                plotfun_slice_mesh(mesh,fun,map,plane)
            elseif length(fun) == size(mesh.m2g,1)
                plotfun_slice_grid(mesh,fun,map,plane)
            else
                error('Plotting function length must match number of nodes or voxels')
            end
            
            % plot inclusion ground truth outline
            if  nargin > 4
                r = incl(:,4);
                c0 = incl(:,1:3);
                sl0 = eval(extractAfter(plane,'='));
                cc = colororder;
                cc = cc(1,:);
                if strcmp(plane(1:2),'x=')
                    for i = 1:length(r)
                        circle_r2 = r(i)^2 - (sl0 - c0(i,1)).^2;
                        if circle_r2 < 0
                            continue
                        end
                        circle_r = sqrt(circle_r2);
                        theta = linspace(0, 2*pi, 100);
                        y = c0(i,2) + circle_r.*cos(theta);
                        z = c0(i,3) + circle_r.*sin(theta);
                        hold on
                        plot(z,y,'LineWidth',1.5,'Color',cc)               
                    end
                elseif strcmp(plane(1:2),'y=')
                    for i = 1:length(r)
                        circle_r2 = r(i)^2 - (sl0 - c0(i,2)).^2;
                        if circle_r2 < 0
                            continue
                        end
                        circle_r = sqrt(circle_r2);
                        theta = linspace(0, 2*pi, 100);
                        x = c0(i,1) + circle_r.*cos(theta);
                        z = c0(i,3) + circle_r.*sin(theta);
                        hold on
                        plot(x,z,'LineWidth',1.5,'Color',cc)
                    end
                elseif strcmp(plane(1:2),'z=')
                    for i = 1:length(r)
                        circle_r2 = r(i)^2 - (sl0 - c0(i,3)).^2;
                        if circle_r2 < 0
                            continue
                        end
                        circle_r = sqrt(circle_r2);
                        theta = linspace(0, 2*pi, 100);
                        x = c0(i,1) + circle_r.*cos(theta);
                        y = c0(i,2) + circle_r.*sin(theta);
                        hold on
                        plot(x,y,'LineWidth',1.5,'Color',cc)
                    end
                end
            end
        end

        % plot slice through mesh, interpoalted to 256 x 256 grid
        function plotfun_slice_mesh(mesh,fun,map,plane)
            % Plots 2D slice of function defined on mesh

            % INPUT
            % mesh = DOGPUP mesh
            % fun = function defined on mesh [NN x 1]
            % map = matlab colormap
            % plane = string that is formated 'x=plane', 'y=plane' or
            % 'z=plane', determines slicing plance
            % incl = inclusion array [NI X (x,y,z) radius]

            % NN = number of nodes, NI = number of inclusions

            % slice function on plane
            [sliceMat,points,plane] = mesh_slice(mesh,plane);
            fun = sliceMat*fun;
            
            res = 512;
            cc = colororder;

            if strcmp(plane(1:2),'x=')
                % plot sliced plane
                y = reshape(points(:,2),res,res);
                z = reshape(points(:,3),res,res);
                fun = reshape(fun,res,res);
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
                xlim([floor(min(mesh.node(:,3))) ceil(max(mesh.node(:,3)))])
                ylim([floor(min(mesh.node(:,2))) ceil(max(mesh.node(:,2)))])

            elseif strcmpi(plane(1:2),'y=')
                % plot sliced plane
                x = reshape(points(:,1),res,res);
                z = reshape(points(:,3),res,res);
                fun = reshape(fun,res,res).';
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
                xlim([floor(min(mesh.node(:,1))) ceil(max(mesh.node(:,1)))])
                ylim([floor(min(mesh.node(:,3))) ceil(max(mesh.node(:,3)))])

            elseif strcmpi(plane(1:2),'z=')
                % plot sliced plane
                x = reshape(points(:,1),res,res);
                y = reshape(points(:,2),res,res);
                fun = reshape(fun,res,res);
                im = imagesc([x(1) x(end)],[y(1) y(end)],fun);
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
                xlim([floor(min(mesh.node(:,1))) ceil(max(mesh.node(:,1)))])
                ylim([floor(min(mesh.node(:,2))) ceil(max(mesh.node(:,2)))])
            end

            colormap(gca,map)
            clim([min(fun(:)) max(fun(:))])
            set(gca,'Color',[0.9 0.9 0.9])
            daspect([1 1 1])

        end
    
        % plot slice through reconstruction basis, no interpolation
        function plotfun_slice_grid(mesh,fun,map,plane)
            % plots slice through function defined on voxel grid
            % see plotfun_slice
            
            res = mesh.gridSize;
            s_pos = str2double(extractAfter(plane,'='));
            x = reshape(mesh.grid(:,1),res(1),res(2),res(3));
            y = reshape(mesh.grid(:,2),res(1),res(2),res(3));
            z = reshape(mesh.grid(:,3),res(1),res(2),res(3));

            fun_plot = NaN.*zeros(size(mesh.grid,1),1);
            fun_plot(mesh.gridinmesh) = fun;
            fun_plot = reshape(fun_plot,res(1),res(2),res(3));

            if strcmp(plane(1:2),'x=')
                % convert slice position to nearest slice index
                x_s = unique(x(:,1,1));
                [~,sIdx] = min(abs(x_s-s_pos));
                % plot sliced plane
                y = pagetranspose(squeeze(y(sIdx,:,:)));
                z = pagetranspose(squeeze(z(sIdx,:,:)));
                fun_plot = squeeze(fun_plot(sIdx,:,:));
                im = imagesc([z(1) z(end)],[y(1) y(end)],fun_plot);
                set(im, 'AlphaData', ~isnan(fun_plot))
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
                fun_plot = pagetranspose(squeeze(fun_plot(:,sIdx,:)));
                im = imagesc([x(1) x(end)],[z(1) z(end)],fun_plot);
                set(im, 'AlphaData', ~isnan(fun_plot))
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
                fun_plot = pagetranspose(squeeze(fun_plot(:,:,sIdx)));
                im = imagesc([x(1) x(end)],[y(1) y(end)],fun_plot);
                set(im, 'AlphaData', ~isnan(fun_plot))
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

            colormap(gca,map)
            set(gca,'Color',[0.9 0.9 0.9])
            daspect([1 1 1])

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
        
        % plot function on 3D mesh
        function plotfun_3d(mesh,fun,map,alpha,selector)
            nodes = mesh.node;
            elems = mesh.elem;
            if nargin > 4
                if ~isletter(selector)  
                    error('Selector must be string or char')
                elseif (isempty(regexp(selector, '[x-zX-Z]', 'once')) && isempty(regexp(selector, '[><=&|]', 'once')))
                    error('Invalid selector expression')
                end
                
                % xyz location of element centre of masses
                x = mean(reshape(nodes(elems.',1),4,[]),1);
                y = mean(reshape(nodes(elems.',2),4,[]),1);
                z = mean(reshape(nodes(elems.',3),4,[]),1);
                % evaluate locations of elements
                selector = lower(string(selector));
                idx = eval(selector);
                % new boundary faces
                faces = [mesh.elem(idx,[1,2,3]);...
                        mesh.elem(idx,[1,2,4]);...
                        mesh.elem(idx,[1,3,4]);...
                        mesh.elem(idx,[2,3,4])];
                % indexes for unique values
                [~,ix,jx]=unique(sort(faces,2),'rows');
                % indexes faces that only appear once i.e are outward facing
                vec = histc(jx,1:max(jx));
                qx = vec == 1;
                faces=faces(ix(qx),:); % connection vectors for surface faces

            else
                faces = mesh.face;
            end

            % plotting limits
            [minPos,maxPos] = bounds(nodes(faces(:),:),'all');
            len = (1.2*maxPos - 1.2*minPos)/2;
            c0 = mean(nodes(faces(:),:),1);
            % plot
            p0 = trimesh(faces,mesh.node(:,3),mesh.node(:,1),mesh.node(:,2));
            p0.EdgeAlpha = alpha/10;
            p0.EdgeColor = 'flat';
            p0.FaceAlpha = alpha;
            p0.FaceVertexCData = fun;
            p0.FaceColor = "interp";
            xlim(c0(3) + [-len len])
            xlim(c0(1) + [-len len])
            xlim(c0(2) + [-len len])
            colormap(map)

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
            set(gca,'XDir','normal','YDir','normal','ZDir','normal')
            view(-45,45)
            axis equal

        end

        % plot isosurface
        function plotfun_iso(mesh,funGT,fun,clrs)
            % plots dice thresholded isosurface of function defined on mesh

            % INPUT
            % mesh = DOGPUP mesh
            % funGT = function defined on mesh [NN x 1]
            % fun = another function defined on mesh [NN x 1]
            % clrs = RGB values for isosurfaces [2 x 3]

            % NN = number of nodes
            
            if nargin < 4 || isempty(clrs)
                clrs = [0.05 0.05 0.05;...
                        1   0   0  ];
            end

            [sliceMat,points] = mesh_slice3(mesh);
            [points,idx] = sortrows([points(:,3),points(:,1),points(:,2)]);
            Z = reshape(points(:,1),128,128,128);
            X = reshape(points(:,2),128,128,128);
            Y = reshape(points(:,3),128,128,128);
            
            plotdmesh(mesh,0.1)
            hold on
            if nargin > 2 && ~isempty(fun)
                fun = sliceMat*fun;
                fun = reshape(fun(idx),128,128,128);
                iso_thresh = 0.5.*(median(fun(fun>0),'all') + max(fun,[],'all'));
                iso = isosurface(Z,X,Y,fun,iso_thresh);
                p = patch(iso);
                set(p,'FaceColor',clrs(2,:),'FaceAlpha',0.4);  
                set(p,'EdgeColor','none');
            end
            
            funGT = sliceMat*funGT;
            funGT = reshape(funGT(idx),128,128,128);
            iso_thresh = 0.5.*(median(funGT(funGT>0),'all') + max(funGT,[],'all'));
            iso = isosurface(Z,X,Y,funGT,iso_thresh);
            p = patch(iso);
            set(p,'FaceColor',clrs(1,:),'FaceAlpha',0.4);
            set(p,'EdgeColor','none');
            daspect([1 1 1])
            grid off
        end
    
    end
end

%% Functions to be called by class methods, not to be called directly

% load json mesh
function [mesh,optode] = load_mesh(fn)
    
    % fn = string or character filename to be read (must be json)

    if ~isletter(fn)
        error('fn must be a string or character')
    end

    [~,~,ext] = fileparts(fn);
    if strcmp(ext,"json")
        error('must be json file')
    end

    % load in json as structure
    structIn = loadjson(fn);
    structIn = structIn.DOGPUP_mesh;


    % initialise mesh object
    mesh = dMesh(structIn.geometry.nodes,...
        structIn.geometry.elements,...
        structIn.parameters.refractiveIndex,...
        structIn.parameters.opticalProps);
    % initialise optode object
    optode = dOptode(structIn.optode.source.position(:,1:3),...
        structIn.optode.detector.position(:,1:3),structIn.optode.link,[],[],[]);
    optode.s_bary = structIn.optode.source.position(:,4:end);
    optode.d_bary = structIn.optode.detector.position(:,4:end);

    % update TD info
    if ~isempty(structIn.optode.timeData)
        t = structIn.optode.timeData(:,1).';
        dt = t(2) - t(1);
        irf = structIn.optode.timeData(:,2:end).';
        optode.tAxis = t;
        optode.dt = dt;
        optode.ch_tirf = irf;
    end

    % update FD info
    if ~isempty(structIn.optode.freqData)
        f = structIn.optode.freqData(:,1).';
        df = f(2) - f(1);
        Nf = length(f);
        irf = structIn.optode.freqData(:,2:end).';
        optode.fAxis = f;
        optode.df = df;
        optode.Nf = Nf;
        optode.ch_firf = irf;
    end

    mesh.optode = optode;

end

% Initialises geometric information such as finding boundary nodes etc.
function mesh = prepare_mesh(mesh)
    
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
    
    % ORIENT FACE ELEMENT NORMAL TO BE OUTWARD FACING AND CALCULATES AREA
    % reorient faces and repair
    [~,face] = meshcheckrepair(mesh.node,mesh.face,'deep');
    % map back to original indexing
    LUT = [unique(face) unique(mesh.face)];
    face = interp1(LUT(:,1),LUT(:,2),face(:));
    mesh.face = reshape(face,size(mesh.face));
    % calculate area
    aVec = mesh.node(mesh.face(:,2),:)-mesh.node(mesh.face(:,1),:);
    bVec = mesh.node(mesh.face(:,3),:)-mesh.node(mesh.face(:,1),:);
    normFace = cross(aVec,bVec,2);
    mesh.area = sqrt(sum(normFace.^2,2)).*0.5;
    
    % FACTOR FOR BOUNDARY CONDITIONS
    % assumes medium-air boundary
    R0 = ((mesh.nr-1).^2)./((mesh.nr+1).^2);
    thetaC = asin(1./mesh.nr);
    A = (2./(1-R0)-1+abs(cos(thetaC)).^3)./(1-abs(cos(thetaC)).^2);
    mesh.R = 1./(2.*A).*mesh.bnd;
    
    % Conversion to integers where needed
    mesh.elem = int32(mesh.elem);
    mesh.face = int32(mesh.face);
    mesh.bnd = int32(mesh.bnd);

end

% generates forward matrices, constucts only upper diagonal
function [K,M] = gen_fwdmat(mesh)
    % Generates upper diagonal of forward matrix components (K,M)
    
    % INPUT
    % MESH = DOGPUP dMesh Object
    
    % OUTPUT
    % K = upper triangle of forward stiffness matrix, attenuation dependant component (n x n)
    % M = upper triangle of forward mass matrix, for frequency dependent / complex component (n x n)
    
    % This version is based on theory from
    % Introduction to finite element methods, Hans Petter Langtangen
    % https://www.uio.no/studier/emner/matnat/ifi/IN5270/h20/ressurser/fem-book-4print.pdf
    % and
    % A gentle introduction to the Finite Element Method, Francisco–Javier Sayas
    % https://team-pancho.github.io/documents/anIntro2FEM_2015.pdf
    % super useful resources

    % Gradient mapping to for stiffness matrix
    idx = sort(mesh.elem,2).';
    idx = idx(:);
    temp = mesh.node(idx,:).';
    temp = reshape(temp,3,4,[]);
    B = temp(:,2:end,:) - temp(:,1,:);
    B = pagetranspose(B);
    B = pagemldivide(B,eye(size(B,1)));
    % Compute forward matrix COO list
    [r,c,k,m] = getMatrix_CUDA(sort(mesh.elem,2),mesh.vol,sort(mesh.face,2),...
        mesh.area,B,mesh.mua,mesh.kappa,mesh.R);
    % Accumulate to MATLAB sparse
    K = sparse(r,c,k,size(mesh.node,1),size(mesh.node,1));
    M = sparse(r(1:length(m)),c(1:length(m)),m,size(mesh.node,1),size(mesh.node,1));

end

% Solve fluence for given source
function phi = fluGPU(mesh,Q)
    % % generate forward matrices
    [K,M] = gen_fwdmat(mesh);
    % generate full forward matrices
    A = K + 1j.*mesh.optode.df./mesh.c.*M;
    A = A + triu(A,1).'; % full fwd matrix
    phi = gpuBicstab_FSAIP(mesh,A,Q,1e-12,1e3); % solves for phiA = nodes x freq x source
end

% BiCGStab with FSAI precon to solve diffusion approx.
function x_out = gpuBicstab_FSAIP(mesh,A,Q,tol,iter)
%GPUBICSTAB uses GPU parallelised bicgstab algo to solve linear system
% assumes Q input is nodes x freqs x sources

%% Initialisation

% generate FSAI preconditioner and CSR of forward matrix
[cPtrG,rPtrG,G] = FSAIP_gen(mesh,A);
[cPtrA,rPtrA,A] = sparse_csr(A);
rGfull = int32(repelem((1:(size(rPtrG,1)-1)).',diff(rPtrG)));
[cPtrGT,rPtrGT,GT] = sparse_csr(cPtrG+1,rGfull,G);
clearvars rGfull

% reshape and get some initial params
len = size(Q,2);
f = (1:len)-1;
Q = complex(permute(Q,[1 3 2])); % [node,source,freq]

% estimate memory needed
if isa(Q,"double")
            datawidth = 8;
elseif isa(Q,'single')
    datawidth = 4;
else
    error('Source must be double or single precision floating point')
end
% memory of source vectors
vram_needed = numel(Q)*datawidth;
% memory from all other intermediate vectors from bicgstab
vram_needed = vram_needed + 12.*vram_needed;
% memory usage of matrices
vram_needed = vram_needed + 2.*numel(G)*datawidth + numel(rPtrG)*4 + numel(cPtrG)*4 +...
     numel(rPtrGT)*4 + numel(cPtrGT)*4 + numel(A)*datawidth + numel(cPtrA)*4 + numel(rPtrA)*4;
vram_needed = vram_needed*2.8;
% due to overheads (MATLAB, drivers, CUDA runtime) we just scale total
% memory a bit as reported avaliable memory is not correct
[~,vram_free] = checkMem_CUDA;
vram_free = vram_free.*0.8;

% batch data depending on memory requirements
n_batch = ceil(vram_needed/vram_free);

if n_batch > 1
    x_out = zeros(size(Q));
    % batch along largest dimension
    n_f = size(Q,3);
    n_s = size(Q,2);
    if n_s > n_f
        n_batch_dim = [min(n_batch,n_s) 1];
    else
        n_batch_dim = [1 min(n_batch,n_f)];
    end
    % check batching
    total_batch = prod(n_batch_dim);
    if total_batch < n_batch
        [~,id] = min(n_batch_dim);
        n_batch_dim(id) = ceil(n_batch/total_batch);
    end

    % check for uneven splits. this can cause data to be cached in
    % standard memory, slowing down GPU solver
    flag = mod([n_s n_f],n_batch_dim) > 0;

    if all(flag) % add extra split to smallest dimension in this case
        [~,id] = min([n_s n_f]);
        n_batch_dim(id) = n_batch_dim(id) + 1;
    else
        n_batch_dim(~flag) = n_batch_dim(~flag) + 1;
    end


    if n_batch_dim(1) > n_s || n_batch_dim(2) > n_f
        warning('Likely not enough memory to solve, solver may be significantly slowed or not execute')
    end
    
    % index of batches
    s_split = floor(linspace(1,n_s,n_batch_dim(1)+1));
    f_split = floor(linspace(1,n_f,n_batch_dim(2)+1));
    
    for i_f = 1:length(f_split)-1
        if i_f == 1
            f_idx = f_split(i_f):f_split(i_f+1);
        else
            f_idx = f_split(i_f)+1:f_split(i_f+1);
        end

        f_temp = f(f_idx);
        G_temp = complex(G(:,f_idx));
        GT_temp = complex(GT(:,f_idx));

        for i_s = 1:length(s_split)-1
            if i_s == 1
                s_idx = s_split(i_s):s_split(i_s+1);
            else
                s_idx = s_split(i_s)+1:s_split(i_s+1);
            end

            Q_temp = complex(Q(:,s_idx,f_idx));
            % send batched problem to GPU
            x_temp = solveField_CUDA(Q_temp,f_temp,rPtrA,cPtrA,A,rPtrG,cPtrG,G_temp,rPtrGT,cPtrGT,GT_temp,tol,int32(iter));
            x_out(:,s_idx,f_idx) = x_temp;
        end
    end
    
else
    % solve all in one
    x_out = solveField_CUDA(Q,f,rPtrA,cPtrA,A,rPtrG,cPtrG,G,rPtrGT,cPtrGT,GT,tol,int32(iter));
end

x_out = permute(x_out,[1 3 2]);


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
    A = tril(A);
    
    N = 30; % do not change unless FSAIP_gen_CUDA.cu is changed accordingly
    
    [i,j,v] = find(A);
    % find 30 (N) max abs vals in each column
    spPattern = cat(2,j,abs(v));
    [spPattern,idx] = sortrows(spPattern,[1 2],'ascend');
    idxMax = diff([spPattern(:,1); spPattern(end,1)+1]);
    idxMax = find(idxMax);
    idxMax = unique(repmat(idxMax,N,1) - kron((0:N-1).',ones(size(idxMax))));
    idxMax = idxMax(idxMax>0);
    idxMax = idx(idxMax); % index of largest abs vals in each row
    spPattern = [i(idxMax),j(idxMax)];
    spPattern = sortrows(spPattern);
    
    %% Convert Sparsity and Forward matrix to CSR
    
    [cPtrG,rPtrG,~] = sparse_csr(spPattern(:,1),spPattern(:,2),ones(size(spPattern,1),1));
    [cPtrA,rPtrA,valA] = sparse_csr(A);
    
    %% Find FSAI for each frequency
    
    valG = FSAIP_gen_CUDA(valA,rPtrA,cPtrA,rPtrG,cPtrG,int32(length(mesh.optode.fAxis))).';
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

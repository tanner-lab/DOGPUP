classdef dOptode < matlab.mixin.Copyable
    % DOGPUP source-detector optode class
    %   Class the defines source detector arrangment
    
    properties (Access = public)
        % sources
        s_dirs % source directions n x 3 where n is number of sources 
        s_positions % source positons mm n x 3
        s_bary % barycentric co-ords of source positions

        % detectors
        d_dirs % detector directions n x 3 where n is number of sources 
        d_positions % positions of detector (mm) n x 3
        d_bary % barycentric co-ords of source positions

        ch_tirf % channel time domain irf
        ch_firf % fourier series coeffs of channel irf

        % time domain stuff
        dt % time spacing (s)
        tAxis % time axis (s)
        % fourier domain stuff
        Nf % number of fourier series coeffs
        df % frequency spacing (s^-1)
        fAxis % frequency axis (s^-1)
        
        link % optode link array
        
    end
    
    methods
        %% Optode Setup
        % Construct an instance of this class
        function optode = dOptode(s_pos,d_pos,link,irf,tAxis,Nf)
            % Construct DOGPUP optode

            % INPUT
            % s_pos = source locations [NS x 3] (mm)
            % d_pos = detector locations [ND x 3] (mm)
            % link = source-detector linking [NM x 2] (source_no, det_no)
            % avgPow = average power of source tpsf
            % tpsf = source tpsf [1 x NT]
            % tAxis = time axis [1 x NT] (s)
            % Nf = number of fourier frequencies

            % OUTPUT
            % optode = fully initialised DOGPUP optode

            % NS = number of sources, ND = number of detectors, 
            % NM number of measurements, NT = number of time steps

            %  Global value setup
            optode.link = link;
            optode.Nf = Nf;
            if ~isempty(tAxis)
                optode.dt = tAxis(2) - tAxis(1);
                optode.tAxis = tAxis;
                T = optode.tAxis(end);
            end
            if ~isempty(Nf)
                w0 = 2*pi/T;
                optode.fAxis = (0:Nf-1).*w0;
                optode.df = optode.fAxis(2) - optode.fAxis(1);
            end

            % Detector setup
            optode.d_positions = d_pos;
            % Source setup
            optode.s_positions = s_pos;
            % IRFs
            optode.ch_tirf = irf;
            % convert to fourier coeffs
            if ~isempty(irf) && ~isempty(Nf) && ~isempty(tAxis)
                optode.ch_firf = td2fc(irf,optode.fAxis,optode.tAxis,2);
            end
        end

        % Snap optodes to mesh
        function snap2mesh(optode,mesh)
            % snap source and detector correctly to mesh

            % INPUT
            % mesh = fully initialised DOGPUP mesh
            % optode = fully initialised DOGPUP optode

            % snap detector to surface
            [optode.d_positions,optode.d_bary,optode.d_dirs] = snap_pos(mesh,optode.d_positions);
            % snap source to surface
            [optode.s_positions,optode.s_bary,optode.s_dirs] = snap_pos(mesh,optode.s_positions);
            % snap source one scattering distance inside
            musp_step = mean(mesh.musp(mesh.elem(optode.s_bary(:,1),:)),2);
            positions_temp = optode.s_positions + 1./musp_step.*optode.s_dirs;
            TR = triangulation(double(mesh.elem),mesh.node);
            [bary_id] = pointLocation(TR,positions_temp);
            % catch misattributed source normals
            id = isnan(bary_id);
            optode.s_dirs(id,:) = -optode.s_dirs(id,:);
            positions_temp = optode.s_positions + 1./musp_step.*optode.s_dirs;
            [bary_id,bary_w] = pointLocation(TR,positions_temp);
            % fix rounding error
            bary_w(bary_w<1e-10) = 0;
            bary_w = bary_w./sum(bary_w,2);
            optode.s_bary = cat(2,bary_id,bary_w);

        end

        %% Optode Update Methods        
        % Update source pulse parameters
        function update_irf(optode,irf,tAxis,Nf)
            % Function to update optode source tpsf

            % OUTPUT
            % optode = optode object with update parameters

            optode.Nf = Nf;
            optode.dt = tAxis(2) - tAxis(1);
            optode.tAxis = tAxis;
            T = optode.tAxis(end);
            w0 = 2*pi/T;
            optode.fAxis = (0:Nf-1).*w0;
            optode.df = optode.fAxis(2) - optode.fAxis(1);

            % IRFs
            optode.ch_tirf = irf;
            % convert to fourier coeffs
            optode.ch_firf = td2fc(irf,optode.fAxis,optode.tAxis,2);
        end
        
        % Update optode positions
        function update_positions(optode,s_pos,d_pos,link)
            % Function to update optode positions
            
            % INPUT
            % optode = DOGPUP optode object
            % s_pos = source locations [NS x 3] (mm)
            % d_pos = detector locations [ND x 3] (mm)
            % link = source-detector linking [NM x 2] (source_no, det_no)

            % OUTPUT
            % optode = optode object with updated positions

            % NS = number of sources, ND = number of detectors, 
            % NM number of measurements
            
            % Update optode positions
            optode.link = link;
            if ~isempty(s_pos)
                % update source position
                optode.s_positions = s_pos;
            end
            if ~isempty(d_pos)
                % update detector position
                optode.d_positions = d_pos;
            end
        end 
   
    end
end

%% Functions to be called by class methods, not to be called directly

function [pos,bary,norm] = snap_pos(mesh,pos)
            
    % Funciton to snap given position to surface of mesh
    % outputs new position in cartesian and barycentric co-ords as
    % well as -ve surface normal
    
    % find closest point to optode
    ids_bnd = dsearchn(mesh.node(mesh.bnd==1,:),pos);
    ids = find(mesh.bnd);
    ids = ids(ids_bnd); % scales indices for all nodes not just boundary
    
    bary = zeros(size(pos,1),5);
    norm = zeros(size(pos));
    
    for i = 1:length(ids)
        % nearest triangles
        face_i = find(any(mesh.face == ids(i),2));
        dist = zeros(size(face_i));
        PP0 = zeros(size(face_i,1),3);
        for ii = 1:length(face_i)
            tri = mesh.node(mesh.face(face_i(ii),:),:);
            [dist(ii),PP0(ii,:)] = pointTriangleDistance(tri,pos(i,:));
        end
        [~,idx] = min(dist);
        % nearest position on triangle
        pos(i,:) = PP0(idx,:);
        aVec = mesh.node(mesh.face(face_i(idx),2),:)-mesh.node(mesh.face(face_i(idx),1),:);
        bVec = mesh.node(mesh.face(face_i(idx),3),:)-mesh.node(mesh.face(face_i(idx),1),:);
        % optode is defined as anti-parallel to intercepting face normal
        normVec = -cross(aVec,bVec,2);
        normVec = normVec./vecnorm(normVec,2,2);
        norm(i,:) = normVec;
        % barycentric co-ord of optode
        TR = triangulation(double(mesh.elem),mesh.node);
        [bary_id,bary_w] = pointLocation(TR,pos(i,:));
        bary(i,:) = cat(2,bary_id,bary_w);
    end
    
        % fix rounding error
        temp_bary = bary(:,2:end);
        temp_bary(temp_bary<1e-12) = 0;
        temp_bary = temp_bary./sum(temp_bary,2);
        bary(:,2:end) = temp_bary;

end

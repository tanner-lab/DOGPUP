classdef dOptode < matlab.mixin.Copyable
    % DOGPUP source-detector optode class
    %   Class the defines source detector arrangment
    
    properties (Access = public)
        % sources
        s_dirs % source directions n x 3 where n is number of sources 
        s_positions % source positons mm n x 3
        s_bary % barycentric co-ords of source positions
        s_avgPow % average power
        s_tpsf % source tpsf
        s_fpsf % fourier series coeffs of fpsf

        % detectors
        d_dirs % detector directions n x 3 where n is number of sources 
        d_positions % positions of detector (mm) n x 3
        d_bary % barycentric co-ords of source positions

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
        function optode = dOptode(s_pos,d_pos,link,avgPow,tpsf,tAxis,Nf)
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
            optode.dt = tAxis(2) - tAxis(1);
            optode.tAxis = tAxis;
            T = optode.tAxis(end);
            w0 = 2*pi/T;
            optode.fAxis = (0:Nf-1).*w0;
            optode.df = optode.fAxis(2) - optode.fAxis(1);

            %   Detector setup
            optode.d_positions = d_pos;
            
            %   Source setup
            optode.s_positions = s_pos;
            % scale tpsf to average power
            optode.s_avgPow = avgPow;
            avgPowTemp = mean(tpsf);
            scaleFact = avgPow/avgPowTemp;
            optode.s_tpsf = tpsf.*scaleFact;
            % convert to fourier coeffs
            optode.s_fpsf = td2fc(optode.s_tpsf,optode.fAxis,optode.tAxis,2);  
        end

        % Snap optodes to mesh
        function optode = snap2mesh(optode,mesh)
            % snap source and detector correctly to mesh

            % INPUT
            % mesh = fully initialised DOGPUP mesh
            % optode = fully initialised DOGPUP optode

            % snap detector to surface
            [optode.d_positions,optode.d_bary,optode.d_dirs] = snap_pos(mesh,optode.d_positions);
            % snap source to surface
            [optode.s_positions,optode.s_bary,optode.s_dirs] = snap_pos(mesh,optode.s_positions);
            % snap source one scattering distance inside
            positions_temp = optode.s_positions + 1/(min(mesh.musp)).*optode.s_dirs;
            TR = triangulation(double(mesh.elem),mesh.node);
            [bary_id,bary_w] = pointLocation(TR,positions_temp);
            % fix rounding error
            bary_w(bary_w<1e-10) = 0;
            bary_w = bary_w./sum(bary_w,2);
            optode.s_bary = cat(2,bary_id,bary_w);

        end

        %% Optode Update Methods        
        % Update source pulse parameters
        function optode = update_tpsf(optode,avgPow,tpsf,tAxis,Nf)
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

            optode.s_avgPow = avgPow;
            avgPowTemp = mean(tpsf);
            scaleFact = avgPow/avgPowTemp;
            optode.s_tpsf = tpsf.*scaleFact;
            optode.s_fpsf = td2fc(optode.s_tpsf,optode.fAxis,optode.tAxis,2);
        end
        
        % Update optode positions
        function optode = update_positions(optode,s_pos,d_pos,link)
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
        temp_bary(temp_bary<1e-10) = 0;
        temp_bary = temp_bary./sum(temp_bary,2);
        bary(:,2:end) = temp_bary;

end

function [dist,PP0] = pointTriangleDistance(TRI,P)
% calculate distance between a point and a triangle in 3D
%
% Copyright (c) 2009, Gwendolyn Fischer
% All rights reserved.
% 
% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are 
% met:
% 
%     * Redistributions of source code must retain the above copyright 
%       notice, this list of conditions and the following disclaimer.
%     * Redistributions in binary form must reproduce the above copyright 
%       notice, this list of conditions and the following disclaimer in 
%       the documentation and/or other materials provided with the distribution
%       
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
% POSSIBILITY OF SUCH DAMAGE.
%
% SYNTAX
%   dist = pointTriangleDistance(TRI,P)
%   [dist,PP0] = pointTriangleDistance(TRI,P)
%
% DESCRIPTION
%   Calculate the distance of a given point P from a triangle TRI.
%   Point P is a row vector of the form 1x3. The triangle is a matrix
%   formed by three rows of points TRI = [P1;P2;P3] each of size 1x3.
%   dist = pointTriangleDistance(TRI,P) returns the distance of the point P
%   to the triangle TRI.
%   [dist,PP0] = pointTriangleDistance(TRI,P) additionally returns the
%   closest point PP0 to P on the triangle TRI.
%
% Author: Gwendolyn Fischer
% Release: 1.0
% Release date: 09/02/02
% Release: 1.1 Fixed Bug because of normalization

% Possible extention could be a version tailored not to return the distance
% and additionally the closest point, but instead return only the closest
% point. Could lead to a small speed gain.

% Example:
% %% The Problem
% P0 = [0.5 -0.3 0.5];
% 
% P1 = [0 -1 0];
% P2 = [1  0 0];
% P3 = [0  0 0];
% 
% vertices = [P1; P2; P3];
% faces = [1 2 3];
% 
% %% The Engine
% [dist,PP0] = pointTriangleDistance([P1;P2;P3],P0);
%
% %% Visualization
% [x,y,z] = sphere(20);
% x = dist*x+P0(1);
% y = dist*y+P0(2);
% z = dist*z+P0(3);
% 
% figure
% hold all
% patch('Vertices',vertices,'Faces',faces,'FaceColor','r','FaceAlpha',0.8);
% plot3(P0(1),P0(2),P0(3),'b*');
% plot3(PP0(1),PP0(2),PP0(3),'*g')
% surf(x,y,z,'FaceColor','b','FaceAlpha',0.3)
% view(3)

% The algorithm is based on 
% "David Eberly, 'Distance Between Point and Triangle in 3D',
% Geometric Tools, LLC, (1999)"
% http:\\www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
%
%        ^t
%  \     |
%   \reg2|
%    \   |
%     \  |
%      \ |
%       \|
%        *P2
%        |\
%        | \
%  reg3  |  \ reg1
%        |   \
%        |reg0\ 
%        |     \ 
%        |      \ P1
% -------*-------*------->s
%        |P0      \ 
%  reg4  | reg5    \ reg6



% rewrite triangle in normal form
B = TRI(1,:);
E0 = TRI(2,:)-B;
E1 = TRI(3,:)-B;


D = B - P;
a = dot(E0,E0);
b = dot(E0,E1);
c = dot(E1,E1);
d = dot(E0,D);
e = dot(E1,D);
f = dot(D,D);

det = a*c - b*b;
s   = b*e - c*d;
t   = b*d - a*e;

% Terible tree of conditionals to determine in which region of the diagram
% shown above the projection of the point into the triangle-plane lies.
if (s+t) <= det
  if s < 0
    if t < 0
      %region4
      if (d < 0)
        t = 0;
        if (-d >= a)
          s = 1;
        else
          s = -d/a;
        end
      else
        s = 0;
        if (e >= 0)
          t = 0;
        else
          if (-e >= c)
            t = 1;
          else
            t = -e/c;
          end
        end
      end %of region 4
    else
      % region 3
      s = 0;
      if e >= 0
        t = 0;
      else
        if -e >= c
          t = 1;
        else
          t = -e/c;
        end
      end
    end %of region 3 
  else
    if t < 0
      % region 5
      t = 0;
      if d >= 0
        s = 0;
      else
        if -d >= a
          s = 1;
        else
          s = -d/a;
        end
      end
    else
      % region 0
      invDet = 1/det;
      s = s*invDet;
      t = t*invDet;
    end
  end
else
  if s < 0
    % region 2
    tmp0 = b + d;
    tmp1 = c + e;
    if tmp1 > tmp0 % minimum on edge s+t=1
      numer = tmp1 - tmp0;
      denom = a - 2*b + c;
      if numer >= denom
        s = 1;
        t = 0;
      else
        s = numer/denom;
        t = 1-s;
      end
    else          % minimum on edge s=0
      s = 0;
      if tmp1 <= 0
        t = 1;
      else
        if e >= 0
          t = 0;
        else
          t = -e/c;
        end
      end
    end %of region 2
  else
    if t < 0
      %region6 
      tmp0 = b + e;
      tmp1 = a + d;
      if (tmp1 > tmp0)
        numer = tmp1 - tmp0;
        denom = a-2*b+c;
        if (numer >= denom)
          t = 1;
          s = 0;
        else
          t = numer/denom;
          s = 1 - t;
        end
      else  
        t = 0;
        if (tmp1 <= 0)
            s = 1;
        else
          if (d >= 0)
              s = 0;
          else
              s = -d/a;
          end
        end
      end
      %end region 6
    else
      % region 1
      numer = c + e - b - d;
      if numer <= 0
        s = 0;
        t = 1;
      else
        denom = a - 2*b + c;
        if numer >= denom
          s = 1;
          t = 0;
        else
          s = numer/denom;
          t = 1-s;
        end
      end %of region 1
    end
  end
end

PP0 = B + s*E0 + t*E1;

dist = norm(P - PP0);
end
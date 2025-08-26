function out = fc2td(funFC,f,t,dim)
% Converts fourier series coefficients to time domain

% INPUT
% funFC = fourier series coefficients
% f = fourier frequencies
% t = time axis
% dim = dimension to integrate along in funFC, should be the dimension that
% corresponds to t
% OUTPUT 
% out = time domain signal for given FC

% error checks
if length(f) ~= size(funFC,dim)
    error('Size along given dimension does not match given fourier coefficients')
elseif dim > ndims(funFC)
    error('Function dimensions and given dimension mismatch')
end

% permute so frequency is in the 1st dimension
permVec = 1:ndims(funFC);
permVec = [permVec(permVec==dim) permVec(permVec~=dim)];
funFC = permute(funFC,permVec);
funFC(1,:,:,:,:,:) = funFC(1,:,:,:,:,:)/2;

% generate transformation matrix
T = exp(1j.*reshape(f,1,[]).*t(:));

% time domain
out = pagemtimes(T,funFC);
out = 2.*real(out);

% put time domain along dim
permVec = 2:ndims(out);
permVec = [permVec(permVec<=dim) 1 permVec(permVec>dim)];
out = permute(out,permVec);
end



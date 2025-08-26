function out = td2fc(funTD,f,t,dim)
% Decomposes to time domain data to Fourier series coefficents

% INPUT
% fun = function to decompose to fourier series
% f = fourier frequencies
% t = time axis
% dim = dimension to integrate along in fun, should be the dimension that
% corresponds to t/f
% OUTPUT 
% out = fourier coefficients for function

% error checks
if length(t) ~= size(funTD,dim)
    error('Size along given dimension does not match given time axis')
elseif dim > ndims(funTD)
    error('Function dimensions and given dimension mismatch')
end

% permute so time is in the 1st dimension
permVec = 1:ndims(funTD);
permVec = [permVec(permVec==dim) permVec(permVec~=dim)];
funTD = permute(funTD,permVec);

% generate transformation matrix
dt = t(2) - t(1);
P = t(end);
T = exp(-1j.*reshape(t,1,[]).*f(:));
% T = T.*cat(2,1/2,ones(1,length(t)-2),1/2);

% fourier coefficients
out = dt.*(pagemtimes(T,funTD))./P;

% put fourier coefficients along dim
permVec = 2:ndims(out);
permVec = [permVec(permVec<=dim) 1 permVec(permVec>dim)];
out = permute(out,permVec);

end


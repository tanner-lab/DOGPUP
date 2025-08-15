function out = fc2tg(funFC,gateFC,f,Nt,dim)
% Converts fourier series coefficients to time gated data

% INPUT
% funFC = fluence fourier series coefficients
% gateFC = timegate fourier series coefficients
% f = fourier frequencies
% t = time axis
% dim = dimension to integrate along in funFC, should be the dimension that
% corresponds to t/f
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
gateFC = conj(gateFC); % assumes time gate x FC dimensionality

% time domain
out = pagemtimes(gateFC,funFC);
out = Nt.*2.*real(out);

% put time domain along dim
permVec = 2:ndims(out);
permVec = [permVec(permVec<=dim) 1 permVec(permVec>dim)];
out = permute(out,permVec);
end



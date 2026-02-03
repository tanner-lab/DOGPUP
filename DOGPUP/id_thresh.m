function id_out = id_thresh(data,thresh,floor)
% Gets index that corresponds to data of interest. If thresh is 'all' this
% is above 1% if thresh is a scalar data is further filter by only
% including data on the falling edge at or below thresh

% INPUT
% data = time domain or time gated fluence [NM x NT]
% thresh = threshold to start indexing on falling edge
% floor = noise floor scalar [0-1]

% OUTPUT
%  id = index of values that are in thresholded region

% default to noise floor as 1% in each channel if no floor given
if nargin < 3 || isempty(floor)
        floor = 0.01;
        data = data./max(data,[],2);
elseif floor > 1 || floor < 0
        error('Noise floor must be defined between 0 (0%) and 1 (100%))')
end


if isstring(thresh) || ischar(thresh)
    if strcmp(thresh,'all')
        data = data./max(data,[],2);
        id_out = data > floor;
    else
        error('thresh must be number or ''all''')
    end
else
    
    % adjust noise floor to either 1% in channel or given floor (relative to most attenuated channels peak count)
    % larger of the two is taken as the floor in each channel
    [~,id_max] = max(data,[],2);
    if nargin > 2
        floor = repmat(floor.*min(max(data,[],2)),size(data,1),1);
        idx = floor<0.01*max(data,[],2);
        floor(idx) = 0.01*max(data(idx,:),[],2);
        thresh = thresh.*max(data,[],2);
    end
    id_incl_strt = repmat(1:size(data,2),size(data,1),1);
    id_incl_end = id_incl_strt;
    [~,id_incl_strt] = max(id_incl_strt >= id_max & data <= thresh,[],2);
    [~,id_incl_end] = max(flip(id_incl_end >= id_max & data > floor,2),[],2);
    id_incl_end = size(data,2) - id_incl_end + 1;

    id_out = false(size(data));
    for i = 1:size(data,1)
        id_out(i,id_incl_strt(i):id_incl_end(i)) = true;
    end
end
end


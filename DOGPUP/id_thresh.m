function id_out = id_thresh(data,thresh)
% Gets index that corresponds to data of interest. If thresh is 'all' this
% is above 1% if thresh is a scalar data is further filter by only
% including data on the falling edge at or below thresh

% INPUT
% data = time domain or time gated fluence [NM x NT]
% thresh = scalar or string

% OUTPUT
%  id = index of values that are in thresholded region


if isstring(thresh) || ischar(thresh)
    if strcmp(thresh,'all')
        data = normalize(data,2,'range');
        id_out = data > 0.01;
    else
        error('thresh must be number or ''all''')
    end
else
    data = normalize(data,2,'range');
    [~,id_max] = max(data,[],2);
    id_incl_strt = repmat(1:size(data,2),size(data,1),1);
    id_incl_end = id_incl_strt;
    id_incl_strt = id_incl_strt >= id_max & data <= thresh;
    id_incl_end = id_incl_end >= id_max & data > 0.01;
    id_out = and(id_incl_strt,id_incl_end);
end
end


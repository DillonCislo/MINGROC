function [P, PI] = pointCloudFarthestPoints(V, k, repelIDx)
%POINTCLOUDFARTHESTPOINTS Use an iterative heuristic to sample a discrete
%set of points so that minimum pairwise distances for each point are
%maximized, i.e. maximize ∑_i min_j ‖pi-pj‖
%
%This is a stripped down version of the 'euclidean' optimization in
%'farthest_points' from 'gptoolbox' by Alec Jacobson using only MATLAB
%built-ins.
%
%   INPUT PARAMETERS:
%
%       - V:            #V x dim list of point cloud coordinates
%       - k:            number of output points
%       - repelIDx:     #RI x 1 vector of repellent indices into V
%
%   OUTPUT PARAMETERS:
%
%       - P:    k x dim list of farthest points sampled from V
%       - PI:   k x 1 vector of indices so that P = V(PI,:)
%
%   by Dillon Cislo 2024/10/24

validateattributes(V, {'numeric'}, {'2d', 'finite', 'real'});
validateattributes(k, {'numeric'}, {'scalar', 'integer', ...
    'nonnegative', 'finite', 'real'});

if (k == 0)
    P = zeros(0, size(V,2));
    PI = [];
    return;
end

if (nargin < 3), repelIDx = []; end
if ~isempty(repelIDx)
    validateattributes(repelIDx, {'numeric'}, {'vector', 'integer', ...
    'postive', 'finite', 'real', '<=', size(V,1)});
    repelIDx = unique(repelIDx);
    if (size(repelIDx, 2) ~= 1), repelIDx = repelIDx.'; end
end

numRepel = numel(repelIDx);
% Ensure that PI does not contain anything in repelIDx
notRepel = setdiff(1:size(V,1), repelIDx)';
notRepel = notRepel(randperm(end));
PI = [notRepel(1:k); repelIDx];

% Embedding of V. Keeping this here in case I ever want to recreate the
% 'biharmonic' funcationality using the point cloud Laplacian from Sharp
% and Crane
EV = V;

% Nearest neighbor IDs/distances from all points to the candidate
% point set
[I, D] = knnsearch(EV(PI,:), EV, 'K', 1);

max_iter = 100;
iter = 1;
while true

    change = false; % Whether to update the candidate point set
    for pi = 1:k

        old_PI_pi = PI(pi); % The current candidate point ID
        if isempty(I)

            % This shouldn't be reachable
            Ipi = true(size(V,1),1);
            J = [1:pi-1, pi+1:k+numRepel];

        else

            Ipi = I==pi;
            J = setdiff(1:numel(PI),pi);
            assert(any(Ipi));

        end

        [IIpi,D(Ipi)] = knnsearch(EV(PI(J),:), EV(Ipi,:), 'K', 1);
        I(Ipi) = J(IIpi);
        fIpi = find(Ipi);
        [~,d] = max(D(fIpi));
        PI(pi) = fIpi(d);
        Dpi = sqrt(sum(bsxfun(@minus,EV(PI(pi),:),EV).^2,2));
        Cpi = Dpi<D;
        D(Cpi) = Dpi(Cpi);
        I(Cpi) = pi;
        change = change || (old_PI_pi ~= PI(pi));
        
    end

    iter = iter + 1;
    if (iter > max_iter)
        warning('Reached max iterations (%d) without convergence', max_iter);
        break;
    end

    if ~change, break; end

end

PI = PI(1:k, :);
P = V(PI, :);

end
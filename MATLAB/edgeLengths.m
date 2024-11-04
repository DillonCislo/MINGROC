function L = edgeLengths(V, F)
%EDGELENGTHS Compute the edge lengths of a mesh triangulation. Basically
%just a rip of 'edge_lengths' from 'gptoolbox' by Alec Jacobson, but only
%using MATLAB built-ins.
%
%   INPUT PARAMETERS:
%
%       - V:    #V x dim set of vertex coordinates
%       - F:    #F x 3 connectivity list
%
%   OUTPUT PARAMETERS:
%
%       - L:    #F by 1 list of edge lengths -OR-
%               #F by 3 list of edge lengths corresponding to [23, 31, 12] -OR-
%               #F by 6 list of edge lengths corresponding to edges
%               opposite *face* pairs: [23 31 12 41 42 43]
%
%   by Dillon Cislo 2024/10/24

% Validate inputs
validateattributes(V, {'numeric'}, {'finite', 'real', '2d'});
validateattributes(F, {'numeric'}, {'finite', 'integer', 'positive', ...
    'real', '2d', '<=', size(V,1)});

switch size(F,2)

    case 2

        L = sqrt(sum((V(F(:,2), :) - V(F(:,1), :)).^2, 2));

    case 3

        L = [ ...
            sqrt(sum((V(F(:,3), :) - V(F(:,2), :)).^2, 2)) ...
            sqrt(sum((V(F(:,1), :) - V(F(:,3), :)).^2, 2)) ...
            sqrt(sum((V(F(:,2), :) - V(F(:,1), :)).^2, 2)) ...
            ];

    case 4

        L = [ ...
            sqrt(sum((V(F(:,4), :) - V(F(:,1), :)).^2, 2)) ...
            sqrt(sum((V(F(:,4), :) - V(F(:,2), :)).^2, 2)) ...
            sqrt(sum((V(F(:,4), :) - V(F(:,3), :)).^2, 2)) ...
            sqrt(sum((V(F(:,2), :) - V(F(:,3), :)).^2, 2)) ...
            sqrt(sum((V(F(:,3), :) - V(F(:,1), :)).^2, 2)) ...
            sqrt(sum((V(F(:,1), :) - V(F(:,2), :)).^2, 2)) ...
            ];

    otherwise

        error('Unsupported simplex size');

end

end


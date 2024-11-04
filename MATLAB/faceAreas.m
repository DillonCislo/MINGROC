function FA = faceAreas(V, F)
%FACEAREAS Computes the area of each face in a mesh triangulation
%
%   INPUT PARAMETERS:
%
%       - V:    #V x dim set of vertex coordinates (2D or 3D)
%       - F:    #F x 3 face connectivity list
%
%   OUTPUT PARAMETERS:
%
%       - FA:    #F by 1 list of face areas
%
%   by Dillon Cislo 2024/10/24

% Validate inputs
validateattributes(V, {'numeric'}, {'finite', 'real', '2d'});
assert(ismember(size(V,2), [2, 3]), ...
    'Vertex coordinates must be either 2D or 3D');
validateattributes(F, {'numeric'}, {'finite', 'integer', 'positive', ...
    'real', '2d', '<=', size(V,1)});

if (size(V,2) == 2)
    V = [V, zeros(size(V,1), 1)];
end

e12 = V(F(:,2), :) - V(F(:,1), :);
e13 = V(F(:,3), :) - V(F(:,1), :);
n = cross(e12, e13, 2);

FA = sqrt(sum(n.^2, 2)) ./ 2;

end


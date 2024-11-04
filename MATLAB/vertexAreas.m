function VA = vertexAreas(V, F)
%VERTEXAREAS Computes the (barycentric) area of each vertex in a mesh
%triangulation. Note that sum(vertexAreas(V, F)) == sum(faceAreas(V, F))
%
%   INPUT PARAMETERS:
%
%       - V:    #V x dim set of vertex coordinates (2D or 3D)
%       - F:    #F x 3 face connectivity list
%
%   OUTPUT PARAMETERS:
%
%       - VA:    #V by 1 list of vertex areas
%
%   by Dillon Cislo 2024/10/24

% Validate inputs
validateattributes(V, {'numeric'}, {'finite', 'real', '2d'});
validateattributes(F, {'numeric'}, {'finite', 'integer', 'positive', ...
    'real', '2d', '<=', size(V,1)});

FA = faceAreas(V, F);
VA = full(sparse(F, 1, repmat(FA, [1 3]), size(V,1), 1)) ./ 3;

end


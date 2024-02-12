function mu = bc_metric(F, x, map)
%BC_METRIC Calculate the Beltrami coefficient of a discrete mapping
%
%   INPUT PARAMETERS:
%
%       - F:            #Fx3 face connectivity list
%
%       - x:            #Vx2 2D undeformed vertex coordinates
%
%       - map:          #VxD transformed vertex coordinates
%
%   OUTPUT PARAMETERS:
%
%       - mu:       #Fx1 complex Beltrami coefficient defined on mesh faces
%
%   by Dillon Cislo 07/20/2020

validateattributes(x, {'numeric'}, {'finite', 'real', 'ncols', 2});
validateattributes(map, {'numeric'}, {'finite', 'real', ...
    'nrows', size(x,1)});
validateattributes(F, {'numeric'}, {'finite', 'integer', 'positive', ...
    'real', 'ncols', 3, '<=', size(x,1)});

op = create_operator(F, x);

if (size(map,2) == 3)
    
    dRdx = op.Dx * map;
    dRdy = op.Dy * map;
    
    E = sum(dRdx.^2, 2);
    G = sum(dRdy.^2, 2);
    F = sum(dRdx .* dRdy, 2);
    
    mu = (E - G + 2i * F) ./ (E + G + 2 * sqrt(E .* G - F.^2));
    
elseif (size(map,2) == 2)
    
    f = complex(map(:,1), map(:,2));
    mu = (op.Dc * f) ./ (op.Dz * f);
    
else
    
    error('Transformed vertex coordinates must be 2D or 3D');
    
end

end


function Op = create_operator(F, V)
%CREATE_OPERATOR A utility function to create a struct that stores a set of
%sparse matrix operators.
%
%   INPUT PARAMETERS:
%
%       - F:        #Fx3 face connectivty list
%
%       - V:        #Vx2 vertex coordinate list
%
%   OUTPUT PARAMETERS:
%
%       - Op:        A struct containing the following sparse operators
%
%           - V2F:      #Fx#V sparse matrix that averages face based
%                       quantities onto vertices
%
%           - F2V:      #Vx#F sparse matrix that averages vertex based
%                       quantities onto faces using angle weights
%
%           - Dx:       #Fx#V sparse matrix operator df/dx
%
%           - Dy:       #Fx#V sparse matrix operator df/dy
%
%           - Dz:       #Fx#V sparse matrix operator df/dz
%
%           - Dc:       #Fx#V sparse matrix operator df/dz*
%
%           - laplacian:    #Vx#V sparse represenation of the
%                           Laplace-Beltrami operator. Equivalent to the
%                           output of 'cotmatrix(x, F)' in gptoolbox (i.e.
%                           symmetric negative semidefinite cotangent
%                           formulation of the weak laplacian)
%
%           - AV:           #Vx#V diagonal sparse matrix. The non-zero
%                           entries are the areas associated to the dual
%                           2-cell of each mesh vertex
%
% by Dillon Cislo 07/30/2020

validateattributes(V, {'numeric'}, {'2d', 'ncols', 2, 'finite', 'real'});
validateattributes(F, {'numeric'}, ...
    {'2d', 'ncols', 3, 'integer', 'positive', 'real', '<=', size(V,1)});

Op = struct();

numF = size(F,1);

%--------------------------------------------------------------------------
% Construct the Mesh Averaging Operators
%--------------------------------------------------------------------------
[Op.V2F, Op.F2V] = meshAveragingOperators(F, V);

%--------------------------------------------------------------------------
% Construct the Mesh Differential Operators
%--------------------------------------------------------------------------

% Row indices of sparse matrix entries
MI = reshape( repmat(1:size(F,1), 3, 1), [3 * size(F,1), 1] );

% Column indices of sparse matrix entries
MJ = reshape( F.', [3 * size(F,1), 1] );

% Extract edge vectors from faces
ei = V(F(:,3), :) - V(F(:,2), :);
ej = V(F(:,1), :) - V(F(:,3), :);
ek = V(F(:,2), :) - V(F(:,1), :);

% Extract edge vector components
eX = [ ei(:,1), ej(:,1), ek(:,1) ].';
eX = eX(:);

eY = [ ei(:,2), ej(:,2), ek(:,2) ].';
eY = eY(:);

% Extract signed double face areas
sdA = ( ei(:,1) .* ej(:,2) - ej(:,1) .* ei(:,2) );
sdA = repmat( sdA, 1, 3 ).';
sdA = sdA(:);

% Construct sparse operator for Dx
MX = -eY ./ sdA;
Op.Dx = sparse( MI, MJ, MX, size(F,1), size(V,1) );

% Construct sparse operator for Dy
MY = eX ./ sdA;
Op.Dy = sparse( MI, MJ, MY, size(F,1), size(V,1) );

% Construct complex operators
Op.Dz = (Op.Dx - 1i .* Op.Dy) ./ 2;
Op.Dc = (Op.Dx + 1i .* Op.Dy) ./ 2;

%--------------------------------------------------------------------------
% Construct the Laplace-Beltrami Operator
%--------------------------------------------------------------------------

% The unsigned double face areas
dblA = abs( ei(:,1) .* ej(:,2) - ej(:,1) .* ei(:,2) );
dblA = repmat(dblA, 1, 3);

% Extract edge lengths from faces
Li = sqrt(sum(ei.^2, 2));
Lj = sqrt(sum(ej.^2, 2));
Lk = sqrt(sum(ek.^2, 2));

% The cotangents of the internal angles of each face
C = [ (Lj.^2 + Lk.^2 - Li.^2), ...
    (Lk.^2 + Li.^2 - Lj.^2), ...
    (Li.^2 + Lj.^2 - Lk.^2) ] ./ dblA ./ 4;

% The bare (unweighted) cotangent Laplace-Beltrami operator
L = sparse(F(:,[2 3 1]), F(:,[3 1 2]), C, size(V,1), size(V,1));
L = L+L';
L = L-diag(sum(L,2));

Op.laplacian = L;

% Construct the weights for the Laplace-Beltrami operator -----------------
% The weights of the operator are simply the inverse total areas of the
% dual vertex 2-cells. These are calculated by determining the contribution
% to the cell from each face adjacent to a vertex and then summing these
% contributions

% Diagonal row/column indices for sparse matrix initialization
I = (1:size(V,1)).';
J = (1:size(V,1)).';

% The contribution to the vertex areas from the attached faces
AF = [Li Lj Lk].^2 .* C;
AF = ( repmat( sum(AF, 2), 1, 3 ) - AF ) ./ 4;
Q = sparse(F, 1, AF, size(V,1), 1);

% The inverse area of each vertex based dual 2-cell
AV = sparse(I, J, Q, size(V,1), size(V,1));

Op.AV = AV;

end


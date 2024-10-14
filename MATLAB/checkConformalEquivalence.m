function conformalError = checkConformalEquivalence(F, V1, V2)
%CHECKCONFORMALEQUIVALENCE A test to determine if two mesh triangulations
%with identical topology are conformally equivalent. Two Euclidean
%triangulations are discretely conformally equivalent if and only if for
%each interior edge, the induced length-cross-ratios are equal. See
%"Discrete conformal maps and ideal hyperbolic polyhedra" by Bobenko,
%Pinkall, and Springborn (2015) for more details.
%
%   NOTE: THIS METHOD ASSUMES THAT ALL FACES ARE CONSISTENTLY CCW ORIENTED
%
%   INPUT PARAMETERS:
%
%       - F:                #Fx3 face connectivity list
%
%       - V1:               #VxD set of vertex coordinates
%
%       - V2:               #VxD set of vertex coordinates
%
%   OUTPUT PARAMETERS:
%
%       - conformalError:	Sum of diffrences of interior edge
%                           length-cross-ratios of the two triangulations
%                           normalized by the number of edges
%
%   by Dillon Cislo 2024/01/25

validateattributes(V1, {'numeric'}, {'2d', 'finite', 'real'});
validateattributes(V2, {'numeric'}, {'2d', 'finite', 'real'});
assert(isequal(size(V1,1), size(V2,1)), 'Improperly sized coordinate sets');

validateattributes(F, {'numeric'}, {'2d', 'ncols', 3, 'finite', ...
    'positive', 'integer', 'real', '<=', size(V1,1)});

TR = triangulation(F,V1);
E = TR.edges;

% Construct edge-face correspondence
resizeCell = @(x) repmat(x, 1, 1+mod(numel(x),2));
efIDx = TR.edgeAttachments(E);
efIDx = cell2mat(cellfun(resizeCell, efIDx, 'Uni', false));

% A list of interior edge IDs
bulkEdgeIDx = diff(efIDx, 1, 2) ~= 0;

% Construct face-edge correspondence
e1IDx = sort( [ F(:,3), F(:,2) ], 2 );
e2IDx = sort( [ F(:,1), F(:,3) ], 2 );
e3IDx = sort( [ F(:,2), F(:,1) ], 2 );

[~, e1IDx] = ismember( e1IDx, sort(E, 2), 'rows' );
[~, e2IDx] = ismember( e2IDx, sort(E, 2), 'rows' );
[~, e3IDx] = ismember( e3IDx, sort(E, 2), 'rows' );

feIDx = [ e1IDx e2IDx e3IDx ];

% Calculate edge lengths
L1 = sqrt(sum((V1(E(:,2),:)-V1(E(:,1),:)).^2, 2));
L2 = sqrt(sum((V2(E(:,2),:)-V2(E(:,1),:)).^2, 2));

% Check that the triangle inequality is satisfied
L1_F = L1(feIDx); L2_F = L2(feIDx);
assert( all(all( (sum(L1_F, 2) - 2 .* L1_F) > 0 )), ...
    'Edges from first mesh do not satisfy the triangle inequality' );
assert( all(all( (sum(L2_F, 2) - 2 .* L2_F) > 0 )), ...
    'Edges from second mesh do not satisfy the triangle inequality' );

% % Calculate face areas from edge lengths
% L1_F = L1(feIDx);
% S1 = sum(L1_F, 2) ./ 2;
% A1_F = sqrt(S1 .* (S1-L1_F(:,1)) .* (S1-L1_F(:,2)) .* (S1-L1_F(:,3)));
% assert( all(A1_F > 0), ...
%     'Edges from first mesh do not satisfy the triangle inequality' );
% 
% L2_F = L2(feIDx);
% S2 = sum(L2_F, 2) ./ 2;
% A2_F = sqrt(S2 .* (S2-L2_F(:,1)) .* (S2-L2_F(:,2)) .* (S2-L2_F(:,3)));
% assert( all(A2_F > 0), ...
%     'Edges from second mesh do not satisfy the triangle inequality' );
%
% % Calculate barycentric edge areas from face areas
% A1_E = zeros(size(E(:,1)));
% A1_E(bulkEdgeIDx) = sum(A1_F(efIDx(bulkEdgeIDx, :)), 2);
% A1_E(~bulkEdgeIDx) = A1_F(efIDx(~bulkEdgeIDx, 1));
% A1_E = A1_E ./ 3;
% 
% A2_E = zeros(size(E(:,1)));
% A2_E(bulkEdgeIDx) = sum(A2_F(efIDx(bulkEdgeIDx, :)), 2);
% A2_E(~bulkEdgeIDx) = A2_F(efIDx(~bulkEdgeIDx, 1));
% A2_E = A2_E ./ 3;

% Calculate edge length-cross-ratios
nextL1 = circshift(L1_F, [0, -1]);
prevL1 = circshift(L1_F, [0, 1]);
lcr1 = accumarray(feIDx(:), nextL1(:), [], @prod) ./ ...
    accumarray(feIDx(:), prevL1(:), [], @prod);
lcr1(~bulkEdgeIDx) = 0;

nextL2 = circshift(L2_F, [0, -1]);
prevL2 = circshift(L2_F, [0, 1]);
lcr2 = accumarray(feIDx(:), nextL2(:), [], @prod) ./ ...
    accumarray(feIDx(:), prevL2(:), [], @prod);
lcr2(~bulkEdgeIDx) = 0;

% Compute the conformal deviation error
conformalError = abs(lcr1-lcr2);
conformalError = sum(conformalError(bulkEdgeIDx)) ./ sum(bulkEdgeIDx);

end


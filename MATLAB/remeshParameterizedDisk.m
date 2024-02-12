function [F, V, x] = remeshParameterizedDisk(F0, V0, x0, varargin)
%REMESHPARAMETERIZEDSURFACE Attempts to remesh a surface with disklike
%topology parameterized in the unit disk to improve quality.
%
%   INPUT PARAMETERS:
%
%       - F0:   #F0x3 input face connectivity list
%
%       - x0:   #V0x2 input 2D vertex coordinate list
%
%       - V0:   #V0x3 input 3D vertex coordinate list
%
%   OPTIONAL INPUT PARAMETERS:
%
%       - ('NewParameterization', newParam = false): Whether or not
%       to generate a new conformal parameterization of the 3D surface in
%       the unit disk using Ricci flow
%
%       - ('TargetLength3D', tarLength3D = []): The target edge length for
%       edge in the 3D surface. If no target length is supplied this is set
%       to the smallest edge length in the input mesh. NOTE: Supplying a
%       non-empty target length will force the generation of a new
%       parameterization
%
%       - ('NumIterations3D', numIter3D = 5): The number of iterations used
%       for CGAL's isotropic remeshing routine.
%
%       - ('ProtectConstraints', protectConstraints = false): Whether or
%       not to protect constraints in CGAL's isotropic remeshing routine
%
%       - ('SmoothVTOL', vtol = 1e-2}: Relative vertex movement tolerance
%       used by the 'smooth2' routine. Smoothing is converged when
%       (VNEW-VERT) <= VTOL * VLEN, where VLEN is a local effective length
%       scale
%
%       - ('NumIterations2D', numIter2D = 32): The number of iterations
%       allowed by the 'smooth2' routine.
%
%       - ('Display', dispType = false): Whether or not to display progress
%
%   OUTPUT PARAMETERS:
%
%       - F:   #Fx3 output face connectivity list
%
%       - x:   #Vx2 output 2D vertex coordinate list
%
%       - V:   #Vx3 output 3D vertex coordinate list

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
validateattributes(V0, {'numeric'}, {'2d', 'finite', 'real', 'ncols', 3});

if ~isempty(x0) 
    
    validateattributes(x0, {'numeric'}, ...
        {'2d', 'finite', 'real', 'ncols', 2});
    
    assert(size(x0,1) == size(V0,1), ...
        'Invalid 2D/3D vertex coordinate input'); 
    
end

validateattributes(F0, {'numeric'}, {'finite', 'integer', 'positive', ...
    'real', 'ncols', 3, '<=', size(V0,1)});

TR0 = triangulation(F0, V0);
E0 = TR0.edges;
bdyIDx0 = freeBoundary(TR0);
bdyIDx0 = bdyIDx0(:,1);

eulerChi = size(V0,1) + size(F0,1) - size(E0,1);
assert(eulerChi == 1, 'Input mesh is not a topological disk');

allBdys = DiscreteRicciFlow.compute_boundaries(F0);
assert(numel(allBdys) == 1, 'Input mesh has multiple boundary components');

% Process Optional Input Parameters ---------------------------------------

newParam = false;
tarLength3D = [];
numIter3D = 5;
protectConstraints = false;
vtol = 1e-2;
numIter2D = 32;
dispType = false;
onePt3D = [];
zeroPt3D = [];

for i = 1:length(varargin)
    
    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'logical'), continue; end
    
    if strcmpi(varargin{i}, 'NewParameterization')
        newParam = varargin{i+1};
        validateattributes(newParam, {'logical'}, {'scalar'});
    end
    
    if strcmpi(varargin{i}, 'TargetLength3D')
        tarLength3D = varargin{i+1};
        if ~isempty(tarLength3D)
            validateattributes(tarLength3D, {'numeric'}, ...
                {'scalar', 'positive', 'finite', 'real'});
        end
    end
    
    if strcmpi(varargin{i}, 'NumIterations3D')
        numIter3D = varargin{i+1};
        validateattributes(numIter3D, {'numeric'}, ...
            {'scalar', 'integer', 'positive', 'finite', 'real'});
    end
    
    if strcmpi(varargin{i}, 'ProtectConstraints')
        protectConstraints = varargin{i+1};
        validateattributes(protectConstraints, {'logical'}, {'scalar'});
    end
    
    if strcmpi(varargin{i}, 'SmoothVTOL')
        vtol = varargin{i+1};
        validateattributes(vtol, {'numeric'}, ...
            {'scalar', 'positive', 'finite', 'real'});
    end

    if strcmpi(varargin{i}, 'NumIterations2D')
        numIter2D = varargin{i+1};
        validateattributes(numIter2D, {'numeric'}, ...
            {'scalar', 'integer', 'positive', 'finite', 'real'});
    end
    
    if strcmpi(varargin{i}, 'Display')
        dispType = varargin{i+1};
        validateattributes(dispType, {'logical'}, {'scalar'});
    end
    
    if strcmpi(varargin{i}, 'OnePoint3D')
        onePt3D = varargin{i+1};
        validateattributes(onePt3D, {'numeric'}, ...
            {'vector', 'numel', 3, 'finite', 'real'});
        if (size(onePt3D, 1) ~= 1), onePt3D = onePt3D.'; end
    end
    
    if strcmpi(varargin{i}, 'ZeroPoint3D')
        zeroPt3D = varargin{i+1};
        validateattributes(zeroPt3D, {'numeric'}, ...
            {'vector', 'numel', 3, 'finite', 'real'});
        if (size(zeroPt3D, 1) ~= 1), zeroPt3D = zeroPt3D.'; end
    end
    
end

%--------------------------------------------------------------------------
% GENERATE A NEW PARAMETERIZATION (IF NECESSARY)
%--------------------------------------------------------------------------
F = F0; V = V0;

if (isempty(x0) || newParam || ~isempty(tarLength3D))
    
    % If no target length is supplied, we set it to the minimum edge length
    % in the 3D mesh
    if isempty(tarLength3D)
        tarLength3D = min(edge_lengths(V0, E0));
    end
    
    % Remesh 3D surface
    [F, V, ~, ~] = isotropic_remeshing(F, V, ...
        tarLength3D, numIter3D, protectConstraints);
    
    % Try to delaunayize the 3D surface
    if ~all(all(is_intrinsic_delaunay(V, F)))
        if dispType
            fprintf('Trying to Delaunayize 3D mesh... ')
        end
        [V, F] = delaunayize(V, F, 'SplitEdges', false);
        if dispType
            fprintf('Done\n')
        end
    end
    
    bdyIDx = freeBoundary(triangulation(F, V));
    bdyIDx = bdyIDx(:,1);
    
    % Generate conformal parameterization in the unit disk
    [~, x, ~] = DiscreteRicciFlow.EuclideanRicciFlow(F, V, ...
        'BoundaryType', 'Fixed', 'BoundaryShape', 'Circles', ...
        'Display', dispType );
    
    if ~isempty(onePt3D)
        oneID = bdyIDx(knnsearch(V(bdyIDx, :), onePt3D));
    elseif ~isempty(x0)
        oneID = bdyIDx0(knnsearch(x0(bdyIDx0, :), [1 0]));
    else
        oneID = bdyIDx(knnsearch(x(bdyIDx, :), [1 0]));
    end
    
    if ~isempty(zeroPt3D)
        zeroID = knnsearch(V, zeroPt3D);
    elseif ~isempty(x0)
        zeroID = knnsearch(x0, [0 0]);
    else
        zeroID = knnsearch(x, [0 0]);
    end

    % Clip boundary points in the domain of parameterization to the unit
    % circle and constrain the Mobius degrees of freedom
    x = complex(x(:,1), x(:,2));
    x(bdyIDx) = exp(1i .* angle(x(bdyIDx)));
    x = (x - x(zeroID)) ./ (1 - conj(x(zeroID)) .* x);
    x = exp(-1i .* angle(x(oneID))) .* x;
    x = [real(x), imag(x)];
    x0 = x; V0 = V;
    
else
    
    x = x0;
    
end

% Find the minimum boundary edge length in the 2D parameterization
minBdyLength = min(edge_lengths(x, freeBoundary(triangulation(F,x))));

% Set natural neighbor interpolation options
nniOptions = struct();
nniOptions.ghostMethod = 2; % 'circle';
nniOptions.GPn = max(500, ceil(2*pi/minBdyLength));
nniOptions.GPr = 2; % Why?
nniOptions.GPe = 1;
nniOptions.gradType = 1; %'direct';

%--------------------------------------------------------------------------
% PERFORM SURFACE REMESHING
%--------------------------------------------------------------------------

% Format the boundary polygon in the necessary way for 'refine2'
bdyIDx = freeBoundary(triangulation(F,x));
bdyIDx = bdyIDx(:,1);

node = x(bdyIDx, :);
edge = (1:size(node,1)).';
edge = [edge, circshift(edge, [-1 0])];

% Generate the edge size constraint function
E = edges(triangulation(F,x));
emp2D = (x(E(:,1), :) + x(E(:,2), :)) ./ 2;
l2D = edge_lengths(x, E);
edgeInterp = scatteredInterpolant(emp2D(:,1), emp2D(:,2), l2D, ...
    'natural', 'linear');
hfun = @(x) edgeInterp(x(:,1), x(:,2));

% Set initial triangulation options
opts = struct();
opts.kind = 'delfront';
opts.rho2 = 1;

[vert, ~, tria, tnum] = refine2(node, edge, [], opts, hfun);

% Set smoothing options
opts = struct();
opts.VTOL = vtol;
opts.ITER = numIter2D;
if dispType
    opts.Iter = 4;
else
    opts.Iter = Inf;
end
[x, ~, F, ~] = smooth2(vert, [], tria, tnum, opts);

% Clip boundary points in the domain of parameterization to the unit
% circle and constrain the Mobius degrees of freedom
bdyIDx = unique(freeBoundary(triangulation(F, x)));
zeroID = knnsearch(x, [0 0]);
oneID = bdyIDx(knnsearch(x(bdyIDx, :), [1 0]));

x = complex(x(:,1), x(:,2));
x(bdyIDx) = exp(1i * angle(x(bdyIDx)));
x = (x - x(zeroID)) ./ (1 - conj(x(zeroID)) .* x);
x = exp(-1i .* angle(x(oneID))) .* x;
x = [real(x), imag(x)];

% Map surface back to 3D
[V, ~, ~] = sibsonInterpolantWithGrad(x0(:,1), x0(:,2), V0, ...
    nniOptions, x(:,1), x(:,2));

end









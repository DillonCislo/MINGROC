function [EG, mu, w, map3D] = computeMINGROC( ...
    F, x, initMap3D, finMap3D, mingrocOptions, nniOptions, varargin)
%COMPUTEMINGROC An optimization procedure to compute the minimum
%information constant growth pattern connecting an initial 3D configuration
%with a final 3D shape
%
%   INPUT PARAMETERS:
%
%       - F:            #Fx3 face connectivity list
%
%       - x:            #Vx2 2D parameterization coordinates
%
%       - initMap3D:    #Vx3 initial 3D surface coordinates. It is assumed
%                       that (x, initMap3D) constitute a conformal
%                       parameterization
%
%       - finMap3D:     #Vx3 final 3D surface coordinates. It is assumed
%                       that (x, finMap3D) constitute a conformal
%                       parameterization
%
%   MINGROC OPTIONS (Default value in parentheses):
%
%       - mingrocOptions.m: The number of corrections to approximate the
%       inverse Hessian matrix in the underlying L-BFGS optimization
%       routine. Values < 3 are not recommended (20)
%
%       - mingrocOptions.epsilon: Absolute tolerance for convergence test.
%       Minimization terminates when the norm of the objective function
%       gradient ||g|| < epsilon (1e-5)
%
%       - mingrocOptions.epsilonRel: Relative tolerance for convergence
%       test. Minimization terminates when the norm of the objective
%       function gradient ||g|| < epsilonRel * ||x|| where ||x|| is the
%       norm of the optimization state (1e-5)
%
%       - mingrocOptions.past: Distance for delta-based convergence test.
%       This parameter determines the distance d to compute the rate of
%       decrease of the objective function, f_{k-d}-f_k, where k is
%       the current iteration step. If this parameter is zero, the
%       delta-based convergence test will not be performed (0)
%
%       - mingrocOptions.delta: Delta for convergence test. Minimization
%       terminates where |f_{k-d}-f_k| < delta * max(1, |f_k|, |f_{k-d}|)
%       (1e-10)
%
%       - mingrocOptions.maxIterations: The maximum number of iterations
%       above which the optimization terminates. Setting this parameter to
%       zero continues an optimization until convergence or error (0)
%
%       - mingrocOptions.minimizationMethod: This parameter specifies
%       whether the minimization is run simultaneously on all terms in the
%       energy or in an alternating fashion with separate optimizations for
%       the terms that depend directly on the Beltrami coefficient and the
%       terms that depend only on the Beltrami coefficient through the
%       quasiconformal mapping. Choices are 'simultaneous' and
%       'alternating' ('simultaneous')
%
%       - mingrocOptions.numGrowthIterations: The number of separate
%       optimization subiterations for the terms that depend only on the
%       Beltrami coefficient through the quasiconformal mapping during
%       alternating optimization (1)
%
%       - mingrocOptions.numMuIterations: The number of separate
%       optimization subiterations for the terms that depend directly on
%       the Beltrami coefficient during alternating optimization (1)
%
%       - mingrocOptions.tCoef: A factor that is multiplied by the average
%       edge length of the mesh squared to determin the short time used to
%       update the Beltrami coefficient during the smoothing steps of the
%       alternating minimization scheme (1)
%
%       - mingrocOptions.lineSearchTermination: The line search termination
%       condition used by the the underlying L-BFGS optimization routine.
%       Choices are 'none' (any step size is accepted), 'decrease'
%       (any step size is accepted, so long as it decreases the energy) or
%       'armijo' (a sufficient decrease requirement) ('decrease')
%
%       - mingrocOptions.maxLineSearch: The maximum number of iterations
%       for the backtracking line search method. Optimization terminates if
%       no valid step size can be found (100)
%
%       - mingrocOptions.minStep: The minimup step length allowed in the
%       line search. This value does not typically need to be modified
%       (1e-20)
%
%       - mingrocOptions.maxStep: The maximum step length allowed in the
%       line search. This value does not typically need to be modified
%       (1e20)
%
%       - mingroc.ftol: A parameter to control the accuracy of the line
%       search routine. This parameter should be greater than 0 and smaller
%       than 0.5 and typically does not need to be modified (1e-4)
%
%       - mingrocOptions.AGC: The coefficient of the term in the discrete
%       energy that seeks to minimize the magnitude of the gradient of the
%       areal growth rates (Area Gradient Coefficient). If this parameter
%       is zero, then this term is not included in the energy (1)
%
%       - mingrocOptions.AC: The coefficient of the term in the discrete
%       energy that seeks to minimize the magnitude of the areal growth
%       rates (Area Coefficient). If this parameter is zero, then this term
%       is not included in the energy (0)
%
%       - mingrocOptions.CC: The coefficient of the term in the discrete
%       energy that seeks to minimize the magnitude of the Beltrami
%       coefficient corresponding to the minimum information constant
%       growth pattern (Conformality Coefficient). If this parameter iz
%       zero, then this term is not included (0)
%
%       - mingrocOptions.SC: The coefficient of the term in the discrete
%       energy that seeks to minimize the magnitude of the gradient of the
%       Beltrami coefficient corresponding to the minimum information
%       constant growth pattern (Smoothness Coefficient). If this parameter
%       is zero then this term is not included in the energy (1).
%
%       - mingrocOptions.DC: The coefficient of the inequality constraint
%       terms in the discrete energy that keeps the magnitude of the
%       Beltrami coefficient corresponding to the minimum information
%       growth pattern less than 1 (Diffeomorphic Coefficient). If this
%       parameter is zero, then this term is not included in the energy (0)
%
%       - mingrocOptions.iterDisp: Whether or not to display
%       textual output for the optimization progress (false)
%
%       - mingrocOptions.iterDispDetailed: Whether or not to display
%       detailed textual output for the optimization progress (false)
%
%       - mingrocOptions.iterVisDisp: Whether or not to display detailed
%       visual output for the optimization progres. NOT YET IMPLEMENTED
%       (false)
%
%       - mingrocOptions.checkSelfIntersections: Whether or not to check
%       for self-intersections in the virtual isothermal parameterization
%       (true)
%
%       - mingrocOptions.recomputeMu: Whether or not to recompute the
%       Beltrami coefficient from the map during each optimization
%       iteration (true)
%
%       - mingrocOptions.smoothMuOnFinalSurface: Whether or not to smooth
%       the Beltrami coefficient on the final 3D surface during the
%       associated updates in the alternating scheme (false)
%
%       - mingrocOptions.use3DEnergy: Whether or not to calculate areas and
%       gradients in the energy with respect to the 2D domain of
%       parameterization of the 3D initial surface (true)
%
%       - mingrocOptions.useVectorSmoothing: Whether or not to treat the
%       Beltrami coefficient as a vector on the 3D surface during the
%       smoothing steps of the alternating scheme (false)
%
%       - mingrocOptions.useVectorEnergy: Whether or not to treat the
%       Beltrami coefficient as a vector on the 3D surface to calculate
%       terms in the energy, i.e. replaces the component-wise scalar
%       Dirichlet energy with a vector Dirichlet energy (false)
%
%   NATURAL NEIGHBOR INTERPLANT OPTIONS (Default value in parenthesis):
%
%       - nniOptions.ghostMethod: The method used for determining the
%       positions of the ghost points used for natural neighbor
%       extrapolation. The choice of method will also dictate the type of
%       gradient estimation used. Choices are 'edge', 'circle', or 'custom'
%       ('edge')
%
%       - nniOptions.customGhostPoints: A set of user defined custom ghost
%       points ([])
%
%       - nniOptions.GPe: The edge length increase factor for edge-based
%       ghost point construction (1)
%
%       - nniOptions.GPr: The radius increase factor of the ghost point
%       circle from the circumcircle of the bounding box (2)
%
%       - nniOptions.GPn: The number of ghost points created using the
%       dense circle method
%
%       - nniOptions.gradType: The method used for gradient generation.
%       Choices are 'direct' or 'iterative'. This choice is overridden if
%       analytic gradients/Hessians are supplied ('direct')
%
%       - nniOptions.DataGradX: #Vx3 matrix. Holds the analytic gradients
%       of the surface 'finMap3D' with respect to the x-component of the 2D
%       parameterization
%
%       - nniOptions.DataGradY: #Vx3 matrix. Holds the analytic gradients
%       of the surface 'finMap3D' with respect to the y-component of the 2D
%       parameterization
%
%       - nniOptions.DataHessXX: #Vx3 matrix. Holds the analytic xx-Hessian
%       of the surface 'finMap3D' with respect to the 2D parameterization
%
%       - nniOptions.DataHessXY: #Vx3 matrix. Holds the analytic xy-Hessian
%       of the surface 'finMap3D' with respect to the 2D parameterization
%
%       - nniOptions.DataHessYY: #Vx3 matrix. Holds the analytic yy-Hessian
%       of the surface 'finMap3D' with respect to the 2D parameterization
%
%   OPTIONAL PARAMETERS (Name, Value)-pairs:
%
%       - ('FixedPointIDx', fixIDx = []): Vertex IDs of points that should
%       be held fixed in the 2D parameterization during optimization
%
%       - ('InitialMap', initW = []): Initial guess for a 2D quasiconformal
%       parameterization corresponding to the minimum information constant
%       growth pattern
%
%       - ('InitialMu', initMu = []): Initial guess for the Beltrami
%       coefficient of the intial map guess. This can just be computed from
%       the initial map.
%
%       - ('InterpMap', interpMap = []): The map used to construct the
%       final surface interpolant. This should constitute a conformal
%       parameterization. Default is x
%
%   OUTPUT PARAMETERS:
%
%       - EG:       The final energy of the minimum information growth
%                   pattern
%
%       - mu:       The Beltrami coefficient corresponding to the minimum
%                   information growth pattern
%
%       - w:        The quasiconformal parameterization corresponding to
%                   the minimum information growth pattern
%
%       - map3D:    The 3D surface coordinates corresponding to the minimum
%                   information growth pattern
%
%   by Dillon Cislo 2024/01/25

%==========================================================================
% INPUT PROCESSING
%==========================================================================
if ((nargin < 5) || isempty(mingrocOptions)), mingrocOptions = struct(); end
if ((nargin < 6) || isempty(nniOptions)), nniOptions = struct(); end

% Process Input Geometry --------------------------------------------------
% Additional checks for topology and quality are performed in the C++ code
validateattributes(x, {'numeric'}, {'2d', 'ncols', 2, 'finite', 'real'});
numV = size(x,1);
assert(max(sqrt(sum(x.^2, 2))) < (1+100*eps), ...
    'Input 2D parameterization does not appear to lie on the unit disk');

validateattributes(initMap3D, {'numeric'}, ...
    {'2d', 'ncols', 3, 'nrows', numV, 'finite', 'real'});

validateattributes(finMap3D, {'numeric'}, ...
    {'2d', 'ncols', 3, 'nrows', numV, 'finite', 'real'});

validateattributes(F, {'numeric'}, {'2d', 'ncols', 3, 'integer', ...
    'positive', 'finite', 'real', '<=', numV});

%**************************************************
% TODO: Implement conformal equivalence checks here
%**************************************************

% Process MINGROC Options -------------------------------------------------
% Additional checks are performed in the C++ code.

if isfield(mingrocOptions, 'minimizationMethod')
    if strcmpi(mingrocOptions.minimizationMethod, 'simultaneous')
        mingrocOptions.minimizationMethod = 1;
    elseif strcmpi(mingrocOptions.minimizationMethod, 'alternating')
        mingrocOptions.minimizationMethod = 2;
    else
        error('Invalid minimization method');
    end
end

if isfield(mingrocOptions, 'lineSearchTermination')
    if strcmpi(mingrocOptions.lineSearchTermination, 'none')
        mingrocOptions.lineSearchTermination = 1;
    elseif strcmpi(mingrocOptions.lineSearchTermination, 'armijo')
        mingrocOptions.lineSearchTermination = 2;
    elseif strcmpi(mingrocOptions.lineSearchTermination, 'decrease')
        mingrocOptions.lineSearchTermination = 3;
    else
        error('Invalid line search termination procedure');
    end
end

if isfield(mingrocOptions, 'iterDisp')
    validateattributes(mingrocOptions.iterDisp, ...
        {'logical'}, {'scalar'});
end

if isfield(mingrocOptions, 'iterDispDetailed')
    validateattributes(mingrocOptions.iterDispDetailed, ...
        {'logical'}, {'scalar'});
end

if isfield(mingrocOptions, 'iterVisDisp')
    validateattributes(mingrocOptions.iterVisDisp, ...
        {'logical'}, {'scalar'});
end

if isfield(mingrocOptions, 'checkSelfIntersections')
    validateattributes(mingrocOptions.checkSelfIntersections, ...
        {'logical'}, {'scalar'});
end

if isfield(mingrocOptions, 'recomputeMu')
    validateattributes(mingrocOptions.recomputeMu, ...
        {'logical'}, {'scalar'});
end

if isfield(mingrocOptions, 'smoothMuOnFinalSurface')
    validateattributes(mingrocOptions.smoothMuOnFinalSurface, ...
        {'logical'}, {'scalar'});
end

if isfield(mingrocOptions, 'use3DEnergy')
    validateattributes(mingrocOptions.use3DEnergy, ...
        {'logical'}, {'scalar'});
end

if isfield(mingrocOptions, 'useVectorSmoothing')
    validateattributes(mingrocOptions.useVectorSmoothing, ...
        {'logical'}, {'scalar'});
end

if isfield(mingrocOptions, 'useVectorEnergy')
    validateattributes(mingrocOptions.useVectorEnergy, ...
        {'logical'}, {'scalar'});
end

% Process NNI Options -----------------------------------------------------
% Additional checks are performed in the C++ code

if isfield(nniOptions, 'ghostMethod')
    if isscalar(nniOptions.ghostMethod)
        assert(ismember(nniOptions.ghostMethod, [1 2 3]), ...
            'Invalid ghost point generation procedure');
    elseif ischar(nniOptions.ghostMethod)
        if strcmpi(nniOptions.ghostMethod, 'custom')
            nniOptions.ghostMethod = 1;
        elseif strcmpi(nniOptions.ghostMethod, 'circle')
            nniOptions.ghostMethod = 2;
        elseif strcmpi(nniOptions.ghostMethod, 'edge')
            nniOptions.ghostMethod = 3;
        else
            error('Invalid ghost point generation procedure');
        end
    else
        error('Invalid ghost point generation procedure');
    end
end

if isfield(nniOptions, 'gradType')
    if isscalar(nniOptions.gradType)
        assert(ismember(nniOptions.gradType, [1 2]), ...
            'Invalid derivative generation procedure');
    elseif ischar(nniOptions.gradType)
        if strcmpi(nniOptions.gradType, 'direct')
            nniOptions.gradType = 1;
        elseif strcmpi(nniOptions.gradType, 'iter')
            nniOptions.gradType = 2;
        else
            error('Invalid derivative generation procedure');
        end
    else
        error('Invalid derivative generation procedure');
    end
end

% Check Optional Inputs ---------------------------------------------------

fixIDx = [];
initMap = [];
initMu = [];
interpMap = [];

for i = 1:length(varargin)
    
    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'logical'), continue; end
    
    if strcmpi(varargin{i}, 'FixedPointIDx')
        fixIDx = varargin{i+1};
        if ~isempty(fixIDx)
            validateattributes(fixIDx, {'numeric'}, {'vector', ...
                'positive', 'real', 'finite', 'integer', '<=', numV});
            if (size(fixIDx,2) ~= 1), fixIDx = fixIDx.'; end
        end
    end
    
    if strcmpi(varargin{i}, 'InitialMap'), initMap = varargin{i+1}; end
    if strcmpi(varargin{i}, 'InitialMu'), initMu = varargin{i+1}; end
    if strcmpi(varargin{i}, 'InterpMap'), interpMap = varargin{i+1}; end
         
end

if (isempty(initMu) && ~isempty(initMap))
    computeMuFromMap = true;
else
    computeMuFromMap = false;
end

if isempty(initMap)
    
    initMap = x;
    
else
    
    if ~isreal(initMap), initMap = [real(initMap), imag(initMap)]; end
    validateattributes(initMap, {'numeric'}, ...
        {'2d', 'finite', 'ncols', 2, 'nrows', numV});
    
    assert(max(sqrt(sum(initMap.^2, 2))) < (1+100*eps), ...
        ['Input 2D quasiconformal parameterization does not appear ' ...
        'to lie on the unit disk']);
end

if computeMuFromMap
    
    initMu = bc_metric(F, x, initMap);
    [~, F2V] = meshAveragingOperators(F, x);
    initMu = F2V * initMu;
    initMu = [real(initMu), imag(initMu)];
    
elseif isempty(initMu)

    initMu = zeros(numV, 2);
    
else
        
    if ~isreal(initMu), initMu = [real(initMu), imag(initMu)]; end
    validateattributes(initMu, {'numeric'}, ...
        {'2d', 'finite', 'ncols', 2, 'nrows', numV});
    
    assert(all(sqrt(sum(initMu.^2, 2)) < 1), ...
        'Invalid input Beltrami coefficient');

end

if isempty(interpMap)
    interpMap = x;
else
    validateattributes(interpMap, {'numeric'}, {'2d', 'ncols', 2, ...
        'finite', 'real', 'nrows', numV});
    assert(max(sqrt(sum(interpMap.^2, 2))) < (1+100*eps), ...
        ['2D parameterization used to construct the final surface ' ...
        'interpolant does not appear to lie on the unit disk']);
end

%==========================================================================
% RUN OPTIMIZATION
%==========================================================================

[EG, mu, w, map3D] = compute_mingroc( F, initMap3D, x, ...
    mingrocOptions, nniOptions, initMu, initMap, finMap3D, ...
    fixIDx, interpMap );

end


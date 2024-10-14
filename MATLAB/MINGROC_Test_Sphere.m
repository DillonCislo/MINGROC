%% MINGROC Test ===========================================================

rmpath(genpath('/home/dillon/Documents/MATLAB/coordinate_transformations'));
addpath(genpath('/home/dillon/Documents/MATLAB/gptoolbox'));
addpath(genpath('/home/dillon/Documents/MATLAB/NaturalNeighborInterpolation/NNI_MATLAB'));
addpath(genpath('/home/dillon/Documents/workspace/NNIpp'));
addpath(genpath('/home/dillon/Documents/MATLAB/GrowSurface'));
addpath(genpath('/home/dillon/Documents/MATLAB/TexturePatch'));
addpath(genpath('/home/dillon/Documents/workspace/MINGROCpp'));
addpath('/home/dillon/Documents/MATLAB/MInGro');
addpath(genpath('/home/dillon/Documents/MATLAB/RicciFlow_MATLAB'));
addpath(genpath('/home/dillon/Documents/MATLAB/CGAL_Code/IsotropicRemeshing/'));
addpath(genpath('/home/dillon/Documents/MATLAB/mesh2d'));
addpath(genpath('/home/dillon/Documents/MATLAB/GeometryCentralCode'));
addpath(genpath('/home/dillon/Documents/MATLAB/tubular/utility/mesh_handling'));

%% Generate 2D Triangulation ==============================================
clear; close all; clc;

diskTri = diskTriangulation(30);
% diskTri = diskTriangulationVogel;
F = bfs_orient(diskTri.ConnectivityList);
x = diskTri.Points;
% E = diskTri.edges;

bdyIDx = diskTri.freeBoundary;
bdyIDx = bdyIDx(:,1);

% Rotate a boundary point to lie at (1,0);
z = complex(x(:,1), x(:,2));
angZ = angle(z(bdyIDx));
rotAng = -min(angZ(angZ > 0));
z = exp(1i * rotAng) .* z;
x = [ real(z), imag(z) ];

% diskTri = triangulation(F, x);

clear z angZ rotAng

%% Generate 3D Triangulation ==============================================

initMap3D = [x, zeros(size(x,1), 1)];

[ finMap3D, gradV, hessV ] = stereographicSphere(x);
finMap3D(:,3) = -finMap3D(:,3);

%% Convert sphere to elliptic paraboloid ----------------------------------

clc;
finMap3D(:,1) = 2 .* finMap3D(:,1);
[F, finMap3D, x] = remeshParameterizedDisk(F, finMap3D, [], ...
    'TargetLength3D', 0.075, 'SmoothVTOL', 5e-3, ...
    'NumIterations2D', 64, 'Display', true, ...
    'OnePoint3D', [2 0 0], 'ZeroPoint3D', [0 0 1]);
initMap3D = [x, zeros(size(x,1), 1)];

if ~all(all(is_intrinsic_delaunay(initMap3D, F)))
    warning('Initial 3D map is NOT intrinsically Delaunay');
end

if ~all(all(is_intrinsic_delaunay(finMap3D, F)))
    warning('Final 3D map is NOT intrinsically Delaunay');
end

if ~all(all(is_intrinsic_delaunay(x, F)))
    warning('2D parameterization is NOT intrinsically Delaunay');
end

diskTri = triangulation(F,x);
E = diskTri.edges;
bdyIDx = diskTri.freeBoundary;
bdyIDx = bdyIDx(:,1);

confMu = bc_metric(F, x, finMap3D);

% View Results ------------------------------------------------------------
figure;

subplot(1,2,1);
triplot(triangulation(F, x));
hold on
scatter([1 0], [0 0], 'filled','r');
hold off
axis equal tight
title('Basic Triangulation');

subplot(1,2,2);
patch('Faces', F, 'Vertices', finMap3D, ...
    'FaceVertexCData', abs(confMu), ...
    'FaceColor', 'flat', 'EdgeColor', 'k');
axis equal tight
colorbar
set(gca, 'Clim', [0 0.5]);


clear confMu


%% Convert sphere to elliptic paraboloid ----------------------------------

finMap3D(:,1) = 2 .* finMap3D(:,1);
[F, finMap3D, ~, ~] = isotropic_remeshing(F, finMap3D, 0.075, 10, false);
% [F, finMap3D, ~, ~] = isotropic_remeshing(F, finMap3D, 0.05, 10, false);

% E = edges(triangulation(F, finMap3D));
bdyIDx = freeBoundary(triangulation(F, finMap3D));
bdyIDx = bdyIDx(:,1);

% Generate conformal parameterization in the unit disk
[~, U, ~] = DiscreteRicciFlow.EuclideanRicciFlow(F, finMap3D, ...
    'BoundaryType', 'Fixed', 'BoundaryShape', 'Circles');

% Clip boundary points in the domain of parameterization to the unit circle
U = complex(U(:,1), U(:,2));
U(bdyIDx) = exp(1i .* angle(U(bdyIDx)));
U = [real(U), imag(U)];

% mobiusIDx = knnsearch( x, [0 0; 1 0] );
mobiusIDx = [ knnsearch(x, [0 0]); knnsearch(finMap3D, [2 0 0]) ];

UC = complex(U(:,1), U(:,2));
w0 = U(mobiusIDx(1), :); w0 = complex(w0(1), w0(2));
UC = (UC - w0) ./ (1 - conj(w0) .* UC);
UC = exp(-1i .* angle(UC(mobiusIDx(2)))) .* UC;

xOld = x;
x = [ real(UC), imag(UC) ];
initMap3D = [x, zeros(size(x,1), 1)];

diskTri = delaunayTriangulation(x);
F = diskTri.ConnectivityList;
E = diskTri.edges;
bdyIDx = diskTri.freeBoundary;
bdyIDx = bdyIDx(:,1);


clear U mobiusIDx UC w0

confMu = bc_metric(F, x, finMap3D);

% View Results ------------------------------------------------------------
figure;

subplot(1,2,1);
triplot(triangulation(F, x));
axis equal tight
title('Basic Triangulation');

subplot(1,2,2);
patch('Faces', F, 'Vertices', finMap3D, ...
    'FaceVertexCData', abs(confMu), ...
    'FaceColor', 'flat', 'EdgeColor', 'k');
axis equal tight
colorbar
set(gca, 'Clim', [0 0.5]);


clear confMu

%% Handle Fixed Point Choices =============================================
close all; clc;

fixIDx = [];
% fixPnts = [-1 0 ];
% fixIDx = knnsearch( x, fixPnts );
% fixIDx = bdyIDx;

triplot(triangulation(F, x));
hold on
scatter( x(fixIDx, 1), x(fixIDx, 2), 'filled', 'r' );
hold off
axis equal tight
title('Fixed Points');

clear fixPnts

%% Set Minimization Options ===============================================
close all; clc;

% Set MINGROC options
mingrocOptions = struct();
mingrocOptions.m = 20;
mingrocOptions.epsilon = 1e-5;
mingrocOptions.epsilonRel = 1e-5;
mingrocOptions.past = 1;
mingrocOptions.delta = 0*1e-10;
mingrocOptions.maxIterations = 100;
mingrocOptions.minimizationMethod = 'simultaneous';
mingrocOptions.numGrowthIterations = 1;
mingrocOptions.numMuIterations = 5;
mingrocOptions.tCoef = 1;
mingrocOptions.lineSearchTermination = 'decrease'; % 'armijo';
mingrocOptions.maxLineSearch = 100;
mingrocOptions.minStep = 1e-20;
mingrocOptions.maxStep = 1e20;
mingrocOptions.ftol = 1e-4;
mingrocOptions.AGC = 1;
mingrocOptions.AC = 0;
mingrocOptions.CC = 0;
mingrocOptions.SC = 1;
mingrocOptions.DC = 0;
mingrocOptions.iterDisp = true;
mingrocOptions.iterDispDetailed = true;
mingrocOptions.checkSelfIntersections = true;
mingrocOptions.recomputeMu = true;
mingrocOptions.useVectorEnergy = false;
mingrocOptions.useVectorSmoothing = false;
mingrocOptions.smoothMuOnFinalSurface = false;
mingrocOptions.use3DEnergy = false;

% Set NNI options
nniOptions = struct();
nniOptions.ghostMethod = 'circle';
nniOptions.GPn = max(500, ...
    ceil(2*pi/min(edge_lengths(x, freeBoundary(triangulation(F,x))))) );
nniOptions.GPr = 2;
nniOptions.GPe = 1;
nniOptions.gradType = 'direct';
% nniOptions.DataGradX = gradV(:,:,1);
% nniOptions.DataGradY = gradV(:,:,2);
% nniOptions.DataHessXX = hessV(:,:,1);
% nniOptions.DataHessXY = hessV(:,:,2);
% nniOptions.DataHessYY = hessV(:,:,3);

%% Run MINGROC ============================================================
close all; clc;

clear EG mu w map3D
[EG, mu, w, map3D] = computeMINGROC(F, x, initMap3D, ...
    finMap3D, mingrocOptions, nniOptions, 'FixedPointIDx', fixIDx);


%% View Results 3D ========================================================
close all; % clc;

figure
% triplot(triangulation(F, [real(w) imag(w)])); axis equal tight

% Calculate Beltrami coefficients
[~, F2V] = meshAveragingOperators(F, x);
finMu3D = F2V * bc_metric(F, x, finMap3D);
mapMu3D = F2V * bc_metric(F, x, map3D);

% Calculate growth rates
dblA0_F = doublearea( initMap3D, F );
dblFinA_F = doublearea( finMap3D, F );
dblA_F = doublearea( map3D, F );

finGamma3D = F2V * (dblFinA_F ./ dblA0_F);
mapGamma3D = F2V * (dblA_F ./ dblA0_F);
% finGamma3D = log(F2V * (dblFinA_F ./ dblA0_F));
% mapGamma3D = log(F2V * (dblA_F ./ dblA0_F));

subplot(2,2,1)
patch( 'Faces', F, 'Vertices', map3D, ...
    'FaceVertexCData', abs(mapMu3D), ...
    'FaceColor', 'interp', 'EdgeColor', 'k' );
axis equal tight
colorbar
set(gca, 'Clim', [0, 0.5]);

subplot(2,2,2)
patch( 'Faces', F, 'Vertices', finMap3D, ...
    'FaceVertexCData', abs(finMu3D), ...
    'FaceColor', 'interp', 'EdgeColor', 'k' );
axis equal tight
colorbar
set(gca, 'Clim', [0, 0.5]);

subplot(2,2,3)
patch( 'Faces', F, 'Vertices', map3D, ...
    'FaceVertexCData', mapGamma3D, ...
    'FaceColor', 'interp', 'EdgeColor', 'k' );
axis equal tight
colorbar
% set(gca, 'Clim', [min(finGamma3D), max(finGamma3D)]);
% set(gca, 'Clim', [0, 0.5]);

subplot(2,2,4)
patch( 'Faces', F, 'Vertices', finMap3D, ...
    'FaceVertexCData', finGamma3D, ...
    'FaceColor', 'interp', 'EdgeColor', 'k' );
axis equal tight
colorbar
set(gca, 'Clim', [min(finGamma3D), max(finGamma3D)]);

%% Vector Smooth the Beltrami Coefficient =================================
close all; clc;

FN = faceNormal(triangulation(F, map3D));
VN = per_vertex_normals(map3D, F, 'Weighting', 'angle');

[V2F, F2V] = meshAveragingOperators(F, map3D);
mu2D_F = bc_metric(F, x, map3D);
mu2D_V = F2V * mu2D_F;

mu3D_F = [real(mu2D_F), imag(mu2D_F)];
mu3D_F = pushVectorField2Dto3DMesh(mu3D_F, x, map3D, F);

[mapV2F, mapF2V] = meshAveragingOperators(F, map3D);
mu3D_V = mapF2V * mu3D_F;
mu3D_V = mu3D_V - dot(mu3D_V, VN, 2) .* VN;

smooth_mu3D_V = vectorDiffuse(F, map3D, mu3D_V, ...
    'timeCoefficient', 1, 'NumIterations', 15, ...
    'DiffusionType', 'trueDiffusion');

smooth_mu3D_F = mapV2F * smooth_mu3D_V;
smooth_mu3D_F = smooth_mu3D_F - dot(smooth_mu3D_F, FN, 2) .* FN;

smooth_mu2D_F = pullVectorField3Dto2DMesh(smooth_mu3D_F, x, map3D, F);
smooth_mu2D_F = complex(smooth_mu2D_F(:,1), smooth_mu2D_F(:,2));
smooth_mu2D_V = F2V * smooth_mu2D_F;

%--------------------------------------------------------------------------
% View Results
%--------------------------------------------------------------------------

COM2D = barycenter(x, F);
COM3D = barycenter(map3D, F);

visVIDx = random_points_on_mesh(map3D, F, 1000);
visFIDx = knnsearch(COM3D, visVIDx);
visVIDx = knnsearch(map3D, visVIDx);

figure

subplot(1,2,1)
patch('Faces', F, 'Vertices', x, 'FaceVertexCData', abs(mu2D_F), ...
    'FaceColor', 'flat', 'EdgeColor', 'k');
hold on
quiver( COM2D(visFIDx,1), COM2D(visFIDx,2), ...
    real(mu2D_F(visFIDx))./abs(mu2D_F(visFIDx)), ... 
    imag(mu2D_F(visFIDx))./abs(mu2D_F(visFIDx)), ...
    1, 'LineWidth', 2, 'Color', 'm');
hold off
axis equal tight
colorbar
title('2D Mu (F)')

subplot(1,2,2)
patch('Faces', F, 'Vertices', x, 'FaceVertexCData', abs(smooth_mu2D_F), ...
    'FaceColor', 'flat', 'EdgeColor', 'k');
hold on
quiver( COM2D(visFIDx,1), COM2D(visFIDx,2), ...
    real(smooth_mu2D_F(visFIDx))./abs(smooth_mu2D_F(visFIDx)), ... 
    imag(smooth_mu2D_F(visFIDx))./abs(smooth_mu2D_F(visFIDx)), ...
    1, 'LineWidth', 2, 'Color', 'm');
hold off
axis equal tight
colorbar
title('Smooth 2D Mu (F)')

figure

subplot(1,2,1)
patch('Faces', F, 'Vertices', map3D, ...
    'FaceVertexCData', normrow(mu3D_F), ...
    'FaceColor', 'flat', 'EdgeColor', 'k');
hold on
quiver3( COM3D(visFIDx,1), COM3D(visFIDx,2), COM3D(visFIDx,3),...
    mu3D_F(visFIDx, 1) ./ normrow(mu3D_F(visFIDx, :)), ... 
    mu3D_F(visFIDx, 2) ./ normrow(mu3D_F(visFIDx, :)), ...
    mu3D_F(visFIDx, 3) ./ normrow(mu3D_F(visFIDx, :)), ...
    1, 'LineWidth', 2, 'Color', 'm');
hold off
axis equal tight
colorbar
title('3D Mu (F)')

subplot(1,2,2)
patch('Faces', F, 'Vertices', map3D, ...
    'FaceVertexCData', normrow(smooth_mu3D_F), ...
    'FaceColor', 'flat', 'EdgeColor', 'k');
hold on
quiver3( COM3D(visFIDx,1), COM3D(visFIDx,2), COM3D(visFIDx,3),...
    smooth_mu3D_F(visFIDx, 1) ./ normrow(smooth_mu3D_F(visFIDx, :)), ... 
    smooth_mu3D_F(visFIDx, 2) ./ normrow(smooth_mu3D_F(visFIDx, :)), ...
    smooth_mu3D_F(visFIDx, 3) ./ normrow(smooth_mu3D_F(visFIDx, :)), ...
    1, 'LineWidth', 2, 'Color', 'm');
hold off
axis equal tight
colorbar
title('Smooth 3D Mu (F)')

figure

subplot(1,2,1)
patch('Faces', F, 'Vertices', map3D, ...
    'FaceVertexCData', normrow(mu3D_V), ...
    'FaceColor', 'interp', 'EdgeColor', 'k');
hold on
quiver3( map3D(visVIDx,1), map3D(visVIDx,2), map3D(visVIDx,3),...
    mu3D_V(visVIDx, 1) ./ normrow(mu3D_V(visVIDx, :)), ... 
    mu3D_V(visVIDx, 2) ./ normrow(mu3D_V(visVIDx, :)), ...
    mu3D_V(visVIDx, 3) ./ normrow(mu3D_V(visVIDx, :)), ...
    1, 'LineWidth', 2, 'Color', 'm');
hold off
axis equal tight
colorbar
set(gca, 'Clim', [min(normrow(mu3D_V)) max(normrow(mu3D_V))]);
title('3D Mu (V)')

subplot(1,2,2)
patch('Faces', F, 'Vertices', map3D, ...
    'FaceVertexCData', normrow(smooth_mu3D_V), ...
    'FaceColor', 'interp', 'EdgeColor', 'k');
hold on
quiver3( map3D(visVIDx,1), map3D(visVIDx,2), map3D(visVIDx,3),...
    smooth_mu3D_V(visVIDx, 1) ./ normrow(smooth_mu3D_V(visVIDx, :)), ... 
    smooth_mu3D_V(visVIDx, 2) ./ normrow(smooth_mu3D_V(visVIDx, :)), ...
    smooth_mu3D_V(visVIDx, 3) ./ normrow(smooth_mu3D_V(visVIDx, :)), ...
    1, 'LineWidth', 2, 'Color', 'm');
hold off
axis equal tight
colorbar
set(gca, 'Clim', [min(normrow(mu3D_V)) max(normrow(mu3D_V))]);
title('Smooth 3D Mu (V)')

%%

figure

subplot(1,2,1)
patch('Faces', F, 'Vertices', x, 'FaceVertexCData', abs(mu2D_V), ...
    'FaceColor', 'interp', 'EdgeColor', 'k');
hold on
quiver( x(visVIDx,1), x(visVIDx,2), ...
    real(mu2D_V(visVIDx))./abs(mu2D_V(visVIDx)), ... 
    imag(mu2D_V(visVIDx))./abs(mu2D_V(visVIDx)), ...
    1, 'LineWidth', 2, 'Color', 'm');
hold off
axis equal tight
colorbar
title('2D Mu (V)')

subplot(1,2,2)
patch('Faces', F, 'Vertices', x, 'FaceVertexCData', abs(smooth_mu2D_V), ...
    'FaceColor', 'interp', 'EdgeColor', 'k');
hold on
quiver( x(visVIDx,1), x(visVIDx,2), ...
    real(smooth_mu2D_V(visVIDx))./abs(smooth_mu2D_V(visVIDx)), ... 
    imag(smooth_mu2D_V(visVIDx))./abs(smooth_mu2D_V(visVIDx)), ...
    1, 'LineWidth', 2, 'Color', 'm');
hold off
axis equal tight
colorbar
title('Smooth 2D Mu (V)')

%%
close all; clc;

fb = freeBoundary(triangulation(F,x));
V = [x, zeros(size(x,1), 1)];

sourceIDx = zeros(size(fb,1), 1);
sourceCoords = zeros(size(fb,1), 3);
for i = 1:size(fb, 1)
    
    progressbar(i, size(fb,1));
    
    sourceIDx(i) = find(any(F == fb(i,1), 2), 1, 'first');
    
    sourceCoords(i,:) = double(F(sourceIDx(i), :) == fb(i,1));
    assert(isequal(unique(sourceCoords(i,:)), [0 1]) && ...
        sum(sourceCoords(i,:)) == 1, 'Bad coords');
    
end

sourceVectors = V(fb(:,2), :) - V(fb(:,1), :);

geodesicPaths = traceGeodesic(F, V, sourceIDx, sourceCoords, sourceVectors);



triplot(triangulation(F,x));

hold on
for i = 1:numel(geodesicPaths)
    quiver( geodesicPaths{i}(1,1), geodesicPaths{i}(1,2), ...
        geodesicPaths{i}(end,1)- geodesicPaths{i}(1,1), ...
        geodesicPaths{i}(end,2)-geodesicPaths{i}(1,2), ...
        0, 'LineWidth', 2, 'Color', 'm', 'MaxHeadSize', 3 );
end
hold off
axis equal tight


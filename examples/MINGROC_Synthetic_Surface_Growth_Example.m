%% MINGROC++ Synthetic Surface Growth Example =============================
% This script walks you through a full example of using MINGROC++ to find
% the "minimum information" constant growth profile for a growing a
% spherical cap into a elliptic paraboloidal cap

% Add the MINGROC++ repository to the PATH
[projectDir, ~, ~] = fileparts(matlab.desktop.editor.getActiveFilename);
addpath(genpath(fullfile(projectDir, '..')));
clear projectDir

%% Load Meshes ============================================================
% A typical use case for MINGRO++ would be where you have two surfaces
% defining the initial and final conditions of a dynamic surface growth
% trajectory. An explanation of the variables:
%
%   - initF:        Face connectivity list for the initial surface
%   - initV:        Vertex coordinate list for the initial surface (3D)
%
%   - F:            Face connectivity list for the final surface
%   - confFinV:     Vertex coordinate list for the final surface (3D)
%                   This is called 'confFinV' since it will store the
%                   conformal 3D coordinates of the surface mesh (relative
%                   to the Lagrangian parameterization) after some
%                   processing below
%
clear; close all; clc;

[projectDir, ~, ~] = fileparts(matlab.desktop.editor.getActiveFilename);
cd(projectDir);
load('Synthetic_Surface_Growth_Example');

% The mathematics of quasiconformal transformations require you to specify
% the motion of one point on the boundary and one point in the bulk (see
% paper for more details), i.e. by tracking the motion of nuclei, in order
% to uniquely determine the mapping in terms of the Beltrami coefficient.
% Therefore these points correspond to the same "material parcel" in both
% surfaces. Here we just pick the 'tip' of the cap and a point on its
% boundary

[~, initZeroID] = max(initV(:,3));
[~, initOneID] = max(initV(:,1));

[~, zeroID] = max(confFinV(:,3));
[~, oneID] = max(confFinV(:,1));

% View Results ------------------------------------------------------------
figure('Color', 'w')

subplot(1,2,1)
trisurf(triangulation(initF, initV));
hold on
scatter3(initV(initZeroID, 1), initV(initZeroID, 2), ...
    initV(initZeroID, 3), 'filled', 'c');
scatter3(initV(initOneID, 1), initV(initOneID, 2), ...
    initV(initOneID, 3), 'filled', 'm');
hold off
axis equal tight
title('Initial Surface')

subplot(1,2,2)
trisurf(triangulation(F, confFinV));
hold on
scatter3(confFinV(zeroID, 1), confFinV(zeroID, 2), ...
    confFinV(zeroID, 3), 'filled', 'c');
scatter3(confFinV(oneID, 1), confFinV(oneID, 2), ...
    confFinV(oneID, 3), 'filled', 'm');
hold off
axis equal tight
title('Final Surface')

%% Remesh Surfaces / Generate Conformal Parameterization ==================
% Frequently meshes in the wild are of poor quality (highly anisotropic
% triangles, low vertex valence, etc.). It is important that your meshes
% are high quality (i.e. triangles are near to equilateral etc.) since they
% will become highly deformed during the optimization process and can
% affect convergence,
%
% We also need to generate conformal parameterizations of each surface in
% the unit disk, which we do using a custom built implementation of the
% Ricci flow for triangle meshes. We will do this all in one fell swoop
% using the helper function 'remeshParameterizedDisk' -- look inside for
% more details about individual steps

remeshOptions = struct();
remeshOptions.tarLength3D = 0.1;
remeshOptions.numIter3D = 15;
remeshOptions.vtol = 1e-2;
remeshOptions.numIter2D = 32;
remeshOptions.dispType = false;

% Generate a parameterization for the initial surface
if ~remeshOptions.dispType
    fprintf('Generating initial time parameterization... ');
end

[initF, initV, initX] = remeshParameterizedDisk(initF, initV, [], ...
    'TargetLength3D', remeshOptions.tarLength3D, ...
    'NumIterations3D', remeshOptions.numIter3D, ...
    'SmoothVTOL', remeshOptions.vtol, ...
    'NumIterations2D', remeshOptions.numIter2D, ...
    'ZeroPoint3D', initV(initZeroID, :), ...
    'OnePoint3D', initV(initOneID, :), ...
    'Display', remeshOptions.dispType );

initZeroID = knnsearch(initX, [0 0]);
initOneID = knnsearch(initX, [1 0]);

if ~remeshOptions.dispType
    fprintf('Done\n');
end

% Generate a parameterization for the final surface
if ~remeshOptions.dispType
    fprintf('Generating final time parameterization... ');
end

[F, confFinV, x] = remeshParameterizedDisk(F, confFinV, [], ...
    'TargetLength3D', remeshOptions.tarLength3D, ...
    'NumIterations3D', remeshOptions.numIter3D, ...
    'SmoothVTOL', remeshOptions.vtol, ...
    'NumIterations2D', remeshOptions.numIter2D, ...
    'ZeroPoint3D', confFinV(zeroID, :), ...
    'OnePoint3D', confFinV(oneID, :), ...
    'Display', remeshOptions.dispType );

zeroID = knnsearch(x, [0 0]);
oneID = knnsearch(x, [1 0]);

if ~remeshOptions.dispType
    fprintf('Done\n');
end

% View Results ------------------------------------------------------------
% The Ricci flow results are basically as conformal as you could possibly
% expect a discrete mapping to be (as determined by a discrete Beltrami
% coeffiecient)
close all; clc;

finRicciMu = bc_metric(F, x, confFinV);
startRicciMu = bc_metric(initF, initX, initV);

figure('Color', 'w')

subplot(2,2,1)
trisurf(triangulation(initF, initV));
hold on
scatter3(initV(initZeroID, 1), initV(initZeroID, 2), ...
    initV(initZeroID, 3), 'filled', 'c');
scatter3(initV(initOneID, 1), initV(initOneID, 2), ...
    initV(initOneID, 3), 'filled', 'm');
hold off
axis equal tight
title('Remeshed Initial Surface (3D)')

subplot(2,2,2)
trisurf(triangulation(F, confFinV));
hold on
scatter3(confFinV(zeroID, 1), confFinV(zeroID, 2), ...
    confFinV(zeroID, 3), 'filled', 'c');
scatter3(confFinV(oneID, 1), confFinV(oneID, 2), ...
    confFinV(oneID, 3), 'filled', 'm');
hold off
axis equal tight
title('Remeshed Final Surface (3D)')

subplot(2,2,3)
patch('Faces', initF, 'Vertices', initX, 'EdgeColor', 'g', ...
    'FaceVertexCData', abs(startRicciMu), 'FaceColor', 'flat');
hold on
scatter(initX(initZeroID, 1), initX(initZeroID, 2), 'filled', 'c');
scatter(initX(initOneID, 1), initX(initOneID, 2), 'filled', 'm');
hold off
axis equal tight
cb1 = colorbar;
cb1.Label.String = '\mid \mu \mid';
set(gca, 'Clim', [0 0.5]);
title('Remeshed Initial Surface (2D)')

subplot(2,2,4)
patch('Faces', F, 'Vertices', x, 'EdgeColor', 'g', ...
    'FaceVertexCData', abs(finRicciMu),'FaceColor', 'flat');
hold on
scatter(x(zeroID, 1), x(zeroID, 2), 'filled', 'c');
scatter(x(oneID, 1), x(oneID, 2), 'filled', 'm');
hold off
axis equal tight
cb2 = colorbar;
cb2.Label.String = '\mid \mu \mid';
set(gca, 'Clim', [0 0.5]);
title('Remeshed Final Surface (3D)')

clear finRicciMu startRicciMu cb1 cb2

%% Re-Parameterize the Initial Surface ====================================
% MINGROC++ currently only works to find growth trajectories between
% surface meshes with identical topologies. We need to remesh the original
% surface to match the final surface topology
close all; clc;

% Options for nearest neighbor interpolation. See '/include/external/NNIpp'
% for more details
nniOptions = struct();
nniOptions.GPn = 500;
nniOptions.GPr = 2;
nniOptions.GPe = 1;
nniOptions.gradType = 1; % 'direct' == 1, 'iter' == 2;
nniOptions.ghostMethod = 2; % 'circle' == 2, 'edge == 3;

% Use nearest neighbor interpolation to re-parameterize the initial surface
initV0 = initV;
[initV, ~, ~] = sibsonInterpolantWithGrad( initX(:,1), initX(:,2), ...
    initV, nniOptions, x(:,1), x(:,2) );

% View Results ------------------------------------------------------------

figure('Color', 'w');

subplot(1,2,1)
trisurf(triangulation(initF, initV0));
hold on
scatter3(initV0(initZeroID, 1), initV0(initZeroID, 2), ...
    initV0(initZeroID, 3), 'filled', 'c');
scatter3(initV0(initOneID, 1), initV0(initOneID, 2), ...
    initV0(initOneID, 3), 'filled', 'm');
hold off
axis equal tight
title('Original Remeshed Surface')

subplot(1,2,2)
trisurf(triangulation(F, initV));
hold on
scatter3(initV(zeroID,1), initV(zeroID,2), ...
    initV(zeroID,3),'filled', 'c');
scatter3(initV(oneID,1), initV(oneID,2), ...
    initV(oneID,3), 'filled', 'm');
hold off
axis equal tight
title('Lagrangian Initial Surface')

%% Run MINGROC++ ==========================================================
close all; clc;

% Set MINGROC options
mingrocOptions = struct();
mingrocOptions.AGC = 1; % The coefficient of the |\nabla \Gamma|^2 term (area growth gradient)
mingrocOptions.AC = 0; % The coefficient of the |\Gamma|^2 term (area growth)
mingrocOptions.SC = 5; % The coefficient of the |\nabla \gamma|^2 term (anisotropy gradient)
mingrocOptions.CC = 0; % The coefficient of the |\gamma|^2 (anisotropy)

% These options are only used in 'alternating' mode. You may be able to
% push the energy slightly lower by playing around, but the overall
% features will remain the same
% mingrocOptions.minimizationMethod = 'alternating';
% mingrocOptions.smoothMuOnFinalSurface = true;
% mingrocOptions.tCoef = 0.05;
% mingrocOptions.numMuIterations = 5;

% Display progress
mingrocOptions.iterDisp = true;

% You can set indices of points whose locations will be fixed throughout
% optimization here. FOr now we leave it blank
fixIDx = [];

[mingroEnergy, mu, w, map3D] = computeMINGROC(F, x, initV, confFinV, ...
    mingrocOptions, nniOptions, 'FixedPointIDx', fixIDx);


%% View Areal Growth Rates ================================================
close all;

initAreas = vertexAreas(initV, F);
confAreas = vertexAreas(confFinV, F);
optiAreas = vertexAreas(map3D, F);

confGamma3D = log(confAreas ./ initAreas);
optiGamma3D = log(optiAreas ./ initAreas);

% The IDs of boundary vertices in the mesh
bdyIDx = freeBoundary(triangulation(F, x));
bdyIDx = [bdyIDx(:,1); bdyIDx(1)];

gamma_crange = [0 6];

figure('Color', 'w', 'Units', 'normalized');
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact'); 

nexttile
trisurf(triangulation(F, confFinV), 'EdgeColor', 'none', ...
    'FaceVertexCData', confGamma3D, 'FaceColor', 'interp');
hold on
plot3(confFinV(bdyIDx,1), confFinV(bdyIDx,2), confFinV(bdyIDx,3), ...
    'LineWidth', 3, 'Color', 'k');
hold off
axis equal
box on
cb1 = colorbar;
cb1.Label.String = 'log(A(T)/A(0))';
cb1.FontWeight = 'bold';
set(gca, 'Clim', gamma_crange);
set(gca, 'Colormap', brewermap(256, 'OrRd'));
xlabel('x'); ylabel('y'); zlabel('z');
title('Conformal Area Growth Rates')

nexttile
trisurf(triangulation(F, map3D), 'EdgeColor', 'none', ...
    'FaceVertexCData', optiGamma3D, 'FaceColor', 'interp');
hold on
plot3(map3D(bdyIDx,1), map3D(bdyIDx,2), map3D(bdyIDx,3), ...
    'LineWidth', 3, 'Color', 'k');
hold off
axis equal
box on
cb2 = colorbar;
cb2.Label.String = 'log(A(T)/A(0))';
cb2.FontWeight = 'bold';
set(gca, 'Clim', gamma_crange);
set(gca, 'Colormap', brewermap(256, 'OrRd'));
xlabel('x'); ylabel('y'); zlabel('z');
title('Optimized Area Growth Rates')

clear cb1 cb2 initAreas confAreas optiAreas
clear confGamma3D optiGamma3D bdyIDx

%% View Ansiotropy ========================================================
% These director field visualization are inherently a little more noisy
% around defects than the streamline plots found in the paper since they do
% not take the magnitude of the Beltrami coefficient into account
% (streamline code relies on additional third-party libraries is not
% included).
close all;

% The IDs of boundary vertices in the mesh
bdyIDx = freeBoundary(triangulation(F, x));
bdyIDx = [bdyIDx(:,1); bdyIDx(1)];

[~, F2V] = meshAveragingOperators(F, x);
muF = bc_metric(F, x, [real(w), imag(w)]); % Beltrami coefficient on faces
muV = F2V * muF; % Beltrami coefficient on vertices

% Anisotropic extension director field on faces in 3D
muDirField = abs(muF) .* exp(1i .* angle(muF) ./ 2);
muDirField = [real(muDirField), imag(muDirField)];
muDirField = pushVectorField2Dto3DMesh(muDirField, x, map3D, F);
muDirField = muDirField ./ sqrt(sum(muDirField.^2, 2));

mu_crange = [0 0.5];
mu_thresh = 2.5e-3;

visFIDx = []; % View direction field on all faces

% Generate a farthest point sampling of faces for direction field
% COM = mean(cat(3, map3D(F(:,1), :), map3D(F(:,2), :), map3D(F(:,3), :)), 3);
% visFIDx = pointCloudFarthestPoints(map3D, 500);
% visFIDx = knnsearch(COM, visFIDx);

% View direction field on all faces above a certain Beltrami threshold
% visFIDx = (1:size(F,1)).';
% visFIDx(abs(muF) < mu_thresh) = [];

patchOptions = {'EdgeColor', 'none', ...
    'FaceVertexCData', abs(muV), 'FaceColor', 'interp'};

plotOptions = {'Color', 'k', 'LineWidth', 1.5};

figure('Color', 'w', 'Units', 'normalized');

plotDirectionField(F, map3D, muDirField, [], ...
    patchOptions, plotOptions, {}, visFIDx);
hold on
plot3(map3D(bdyIDx,1), map3D(bdyIDx,2), map3D(bdyIDx,3), ...
    'LineWidth', 3, 'Color', 'k');
hold off
axis equal
box on
cb = colorbar;
cb.Label.String = '\mid \mu \mid';
cb.FontWeight = 'bold';
set(gca, 'Clim', mu_crange);
set(gca, 'Colormap', brewermap(256, 'YlGnBu'));
xlabel('x'); ylabel('y'); zlabel('z');
title('Optimized Anisotropy')

clear cb F2V muV muF muDirField patchOptions plotOptions visFIDx bdyIDx V2F

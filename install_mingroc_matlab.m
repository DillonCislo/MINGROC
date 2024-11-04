%% INSTALL MINGROC++ FOR MATLAB ===========================================
% Make sure you follow the instructions in the README to install
% third-party dependencies not included in this repository first.
% After that, most Linux/Mac users can just hit 'Run'!
clear; close all; clc;

disp('*************************************');
disp('INSTALLING MINGROC++ FOR MATLAB');
disp('*************************************');

% Set up basic directory structure
[mingrocDir, ~, ~] = fileparts(matlab.desktop.editor.getActiveFilename);
extDir = fullfile(mingrocDir, 'include', 'external');
nniDir = fullfile(extDir, 'NNIpp', 'MATLAB', 'mex');
mexDir = fullfile(mingrocDir, 'MATLAB', 'mex');

% Set compilation options - if you want to point to non-default dependency
% locations, set the appropriate PATH variables here!!
compilationOptions = struct();

% Verbose output
compilationOptions.verbose = false;

% Recompile everything, even if it already exists
compilationOptions.compileFromScratch = true;

% Set paths to required libraries
compilationOptions.eigenDir = fullfile(extDir, 'eigen-3.4.0');
compilationOptions.CGALDir = fullfile(extDir, 'CGAL-6.0.1', 'include');
compilationOptions.iglDir = fullfile(extDir, 'libigl', 'include');
compilationOptions.iglPredDir = fullfile(extDir, 'libigl-predicates');

% % If you have non-default Boost or TBB installations set them here
% % The compilation subfunctions will search the default locations
% compilationOptions.boostDir = '/usr/include/boost';
% compilationOptions.tbbDir = '/usr/include/tbb';

% Unzip CGAL and Eigen libraries
cd(extDir);
fprintf('Unzipping CGAL and Eigen... ')
if ~(strcmp(compilationOptions.CGALDir, fullfile(extDir, 'CGAL-6.0.1', 'include')) && ...
    strcmp(compilationOptions.eigenDir, fullfile(extDir, 'eigen-3.4.0')))
    choice = questdlg(['Non-default CGAL and Eigen directories detected. ' ...
    'Do you want to unzip the contents?'], 'Unzip Confirmation', ...
    'Yes', 'No', 'No');
    switch choice
    case 'Yes'
        unzip_cgal_and_eigen(compilationOptions.verbose);
        fprintf('Done.\n')
    case 'No'
        disp('Skipping unzipping.')
    end
else
    unzip_cgal_and_eigen(compilationOptions.verbose);
    fprintf('Done.\n')
end

% Compile LibIGL Predicates
cd(extDir);
fprintf('Compiling libigl predicates... ')
compile_libigl_predicates(compilationOptions);
fprintf('Done.\n\n')

% Compile CGAL function mex bindings
cd(mexDir);
alreadyCompiled = isfile(fullfile(mexDir, ['isotropic_remeshing.' mexext]));
if ~alreadyCompiled || compilationOptions.compileFromScratch

    disp('Compiling isotropic remeshing:')
    compile_isotropic_remeshing(compilationOptions);
    disp(' ')

else

    disp('Isotropic remeshing is already compiled.');
    disp(' ');

end

% Compile NNIpp mex files
cd(nniDir);
alreadyCompiled = isfile(fullfile(nniDir, ['sibsonInterpolant.' mexext])) && ...
    isfile(fullfile(nniDir, ['sibsonInterpolantWithGrad.' mexext]));
if ~alreadyCompiled || compilationOptions.compileFromScratch

    disp('Compiling NNIpp:' );
    compile_sibson_interpolant(compilationOptions);
    disp(' ');

    disp('Compiling NNIpp with gradient:' );
    compile_sibson_interpolant_with_grad(compilationOptions);
    disp(' ');

else

    disp('NNIpp is already compiled.');
    disp(' ');

end

% Compile MINGROC++ mex files
cd(mexDir);
compilationOptions.compileFromScratch = true;
alreadyCompiled = isfile(fullfile(mexDir, ['compute_mingroc.' mexext])) && ...
    isfile(fullfile(mexDir, ['compute_mingroc_energy.' mexext]));
if ~alreadyCompiled || compilationOptions.compileFromScratch

    disp('Compiling MINGROC:' );
    compile_mingroc(compilationOptions);
    disp(' ');

    disp('Compiling MINGROC energy computation:' );
    compile_compute_mingroc_energy(compilationOptions);
    disp(' ');

else

    disp('MINGROC is already compiled.');
    disp(' ');

end

cd(mingrocDir);
disp('All done!');
disp('*************************************');

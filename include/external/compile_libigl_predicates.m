function compile_libigl_predicates(compilationOptions)

    if (nargin < 1)
        compilationOptions = struct();
    end

    if isfield(compilationOptions, 'verbose')
        verbose = compilationOptions.verbose;
    else
        verbose = false;
    end

    if isfield(compilationOptions, 'compileFromScratch')
        compileFromScratch = compilationOptions.compileFromScratch;
    else
        compileFromScratch = false;
    end

    % Get the directory of the current file
    projectDir = fileparts(mfilename('fullpath'));
    cd(projectDir)

    % Define the build directory
    buildDir = fullfile(projectDir, 'libigl-predicates', 'build');

    % Check if the library is already compiled
    alreadyCompiled = isfolder(buildDir) && ...
        isfile(fullfile(buildDir, 'libigl-predicates.a'));
    if alreadyCompiled && ~compileFromScratch
        if verbose
            disp('libigl-predicates is already compiled.');
        end
        return;
    end

    % Check if the source directory exists
    if exist(buildDir, 'dir')
        if compileFromScratch
            rmdir(buildDir, 's');
            mkdir(buildDir);
        end
    else
        mkdir(buildDir);
    end
    
    % Change to the build directory
    cd(buildDir);
    
    % Run cmake and make commands
    if verbose
        cmakeCommand = 'cmake ../';
        makeCommand = 'make';
    else
        cmakeCommand = 'cmake ../ > /dev/null 2>&1';
        makeCommand = 'make > /dev/null 2>&1';
    end
    
    system(cmakeCommand);
    status = system(makeCommand);
    if status ~= 0
        error('An error occurred while compiling libigl-predicates.');
    else
        if verbose, disp('libigl-predicates compiled successfully.'); end
    end
    % Change back to the original directory
    cd(projectDir);

end
function unzip_cgal_and_eigen(verbose)

if (nargin < 1), verbose = false; end

[projectDir, ~, ~] = fileparts(mfilename('fullpath'));
cd(projectDir);

% Define paths for tarballs
cgalTarball = fullfile(projectDir, 'CGAL-6.0.1.tar.xz');
eigenTarball = fullfile(projectDir, 'eigen-3.4.0.tar.gz');

% Define the expected directory names after extraction
cgalDir = fullfile(projectDir, 'CGAL-6.0.1');
eigenDir = fullfile(projectDir, 'eigen-3.4.0');

% Extract CGAL source code, if necessary
if ~isfolder(cgalDir)
    if verbose
        fprintf('CGAL directory not found. Extracting CGAL from %s...\n', cgalTarball);
    end
    try
        % Use a system command to extract the .tar.xz file
        system(['tar -xf ', cgalTarball, ' -C ', projectDir]);
        if verbose
            fprintf('CGAL successfully extracted.\n');
        end
    catch ME
        fprintf('Error extracting CGAL: %s\n', ME.message);
    end
else
    if verbose
        fprintf('CGAL is already extracted.\n');
    end
end

% Extract Eigen source code if necessary
if ~isfolder(eigenDir)
    if verbose
        fprintf('Eigen directory not found. Extracting Eigen from %s...\n', eigenTarball);
    end
    try
        untar(eigenTarball, projectDir);
        if verbose
            fprintf('Eigen successfully extracted.\n');
        end
    catch ME
        fprintf('Error extracting Eigen: %s\n', ME.message);
    end
else
    if verbose
        fprintf('Eigen is already extracted.\n');
    end
end

end
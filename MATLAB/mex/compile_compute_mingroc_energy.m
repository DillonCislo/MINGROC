function compile_compute_mingroc_energy(compilationOptions)

% Validate input: Check if required fields exist in the struct
if (nargin < 1)
    compilationOptions = struct();
end

% Eigen directory (required), e.g. '/usr/local/include/eigen-3.4.0'
if ~isfield(compilationOptions, 'eigenDir') || isempty(compilationOptions.eigenDir)
    error('Please supply Eigen path in compilationOptions.eigenDir');
end

% LibIGL directory (required), e.g. '/usr/local/include/libigl/include'
if ~isfield(compilationOptions, 'iglDir') || isempty(compilationOptions.iglDir)
    error('Please supply LibIGL path in compilationOptions.iglDir');
end

% LibIGL-Predicates directory (required), e.g. '/usr/local/include/libigl-predicates'
if ~isfield(compilationOptions, 'iglPredDir') || isempty(compilationOptions.iglPredDir)
    error('Please supply LibIGL-Predicates path in compilationOptions.iglPredDir');
end

% CGAL directory (required), e.g. '/usr/local/include/CGAL-6.0/include'
if ~isfield(compilationOptions, 'CGALDir') || isempty(compilationOptions.CGALDir)
    error('Please supply CGAL path in compilationOptions.CGALDir');
end

% Boost directory (optional, defaults to system locations)
if ~isfield(compilationOptions, 'boostDir') || isempty(compilationOptions.boostDir)
    if isfolder('/usr/include/boost')
        boostDir = '/usr/include/boost';
    elseif isfolder('/usr/local/include/boost')
        boostDir = '/usr/local/include/boost';
    else
        error('Boost library not found in default locations');
    end
else
    boostDir = compilationOptions.boostDir;
end

% TBB directory (optional, defaults to system locations)
if ~isfield(compilationOptions, 'tbbDir') || isempty(compilationOptions.tbbDir)
    if isfolder('/usr/include/tbb')
        tbbDir = '/usr/include/tbb';
    elseif isfolder('/usr/local/include/tbb')
        tbbDir = '/usr/local/include/tbb';
    else
        error('TBB library not found in default locations');
    end
else
    tbbDir = compilationOptions.tbbDir;
end

[projectDir, ~, ~] = fileparts(mfilename('fullpath'));
cd(projectDir);

CXXOPTIMFLAGS = '"-O3" ';
CXXFLAGS = '"$CXXFLAGS -march=native -fopenmp -fPIC -std=c++17" ';
LDFLAGS = '"$LDFLAGS -fopenmp" ';
CGALCompileFlags = '-DCGAL_EIGEN3_ENABLED=true ';

includeFlags = [ ...
    '-I' compilationOptions.eigenDir ' ' ...
    '-I' compilationOptions.iglDir ' ' ...
    '-I' compilationOptions.iglPredDir ' ' ...
    '-I' boostDir ' ' ...
    '-I' compilationOptions.CGALDir ' ' ...
    '-I' tbbDir ' ' ...
    '-I/usr/include:/usr/local/include ' ];

libFlags = '-L/usr/lib:/usr/local/lib:/usr/lib/x86-64-linux-gnu ';
libFlags = [libFlags '-L' compilationOptions.iglPredDir '/build '];
libFlags = [ libFlags '-lgmp -lmpfr -lboost_thread -lboost_system -ltbb -lpredicates' ];

if compilationOptions.verbose

    mexString = [ 'mex -R2018a -v -O compute_mingroc_energy.cpp ' ...
        'CXXOPTIMFLAGS=' CXXOPTIMFLAGS ' ' ...
        'CXXFLAGS=' CXXFLAGS ' ' ...
        'LDFLAGS=' LDFLAGS ' ' ...
        CGALCompileFlags includeFlags libFlags ];

else

    mexString = [ 'mex -R2018a -O compute_mingroc_energy.cpp ' ...
        'CXXOPTIMFLAGS=' CXXOPTIMFLAGS ' ' ...
        'CXXFLAGS=' CXXFLAGS ' ' ...
        'LDFLAGS=' LDFLAGS ' ' ...
        CGALCompileFlags includeFlags libFlags ];

end

eval(mexString);

end
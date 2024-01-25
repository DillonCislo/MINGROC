function compile_mingroc
    
clc;

[projectDir, ~, ~] = fileparts(matlab.desktop.editor.getActiveFilename);
cd(projectDir);

CGALDir = '/usr/local/include/CGAL-5.6/include';
boostDir = '/usr/include/boost';
eigenDir = '/usr/local/include/eigen-3.4.0';
iglDir = '/usr/local/include/libigl/include';
tbbDir = '/usr/include/tbb';

CXXOPTIMFLAGS = '"-O3" ';
CXXFLAGS = '"$CXXFLAGS -march=native -fopenmp -fPIC -std=c++17" ';
LDFLAGS = '"$LDFLAGS -fopenmp" ';
CGALCompileFlags = '-DCGAL_EIGEN3_ENABLED=true ';

includeFlags = [ ...
    '-I' eigenDir ' ' ...
    '-I' iglDir ' ' ...
    '-I' boostDir ' ' ...
    '-I' CGALDir ' ' ...
    '-I' tbbDir ' ' ...
    '-I/usr/include:/usr/local/include ' ];

libFlags = '-L/usr/lib:/usr/local/lib:/usr/lib/x86-64-linux-gnu ';
% libFlags = [libFlags '-L' tbbDir '/../lib '];
libFlags = [ libFlags '-lgmp -lmpfr -lboost_thread -lboost_system -ltbb' ];

mexString = [ 'mex -R2018a -v -O compute_mingroc.cpp ' ...
    'CXXOPTIMFLAGS=' CXXOPTIMFLAGS ' ' ...
    'CXXFLAGS=' CXXFLAGS ' ' ...
    'LDFLAGS=' LDFLAGS ' ' ...
    CGALCompileFlags includeFlags libFlags ];

eval(mexString);

end
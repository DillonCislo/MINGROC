# MINGROC++
Compute a "minimum information" constant growth pattern between two surfaces. A C++/MATLAB implementation of the algorithm described in ['A "morphogenetic action" principle for 3D shape formation by the growth of thin sheets'](https://arxiv.org/abs/2302.07839) by Dillon Cislo, Anastasios Pavlopoulos, and Boris Shraiman. The paper is currently under review at PRX and is substantially improved over the current version posted to the arXiv. Check back soon for updates on both the paper and the associated code!

## Computing "Optimal" Growth Trajectories

The paper develops a theoretical and computational framework that allows for the quantitative characterization of different developmental trajectories of two-dimensional tissues forming three-dimensional structures. Using this framework, the problem of growth pattern selection can be framed as an optimization problem and solved for optimal growth patterns linking initial and target shapes. In particular we explore the simple optimality principle:

$$\mathcal{S} = \lambda \, D_{SEM} + \int_0^T dt \, \int_{\mathcal{B}} dA(t) \, \, \left[ c_1 \, |\nabla {\Gamma}|^2 + c_2 \, |\nabla \gamma|^2 + c_3 \, \dot{\Gamma}^2 + c_4 \, |\dot{\gamma}|^2 \right],$$

where $\Gamma$ is the relative rate of local area growth and $\gamma$ is the relative rate of change of local deformation anisotropy. The first term is an enpoint cost that ensures the growth pattern ends up generating the correct target shape. This is a really hard computational problem to solve so, for simplicity, we focus on the special case of "constant growth", i.e. $\dot{\Gamma} = \dot{\gamma} = 0$. After some additional simplifications we arrive at the modified constant growth functional:

$$\mathcal{S} \sim \int_{\mathcal{B}} dA(t = 0) \, \, \left[ c_1 \, |\nabla {\Gamma}|^2 + c_2 \,|\nabla \gamma|^2 \right].$$

Basically we are finding the smoothest pattern than can be coded into the tissue at the start of the growth process and then let run to generate the final shape. For constant growth patterns we can also ensure that the target shape constraint is satisfied by construction without an explicit endpoint cost. For more details please see the paper, specifically Section III and Appendix B.

## Installation
`MINGROC++` relies on a few third-party software packages, notably [libigl](https://libigl.github.io/), [CGAL](https://www.cgal.org/), and [Eigen](https://eigen.tuxfamily.org/). Most of these packages are neatly included within this repository, either as submodules or as source tarballs. You are required to install a small set of packages on your own: [CMake](https://cmake.org/), [Boost](https://www.boost.org/), [TBB](https://www.threadingbuildingblocks.org/), [GMP](https://gmplib.org/), and [MPFR](https://www.mpfr.org/). 

#### Linux
```sh
sudo apt-get update
sudo apt-get install cmake libboost-all-dev libtbb-dev libgmp-dev libmpfr-dev
```

#### Mac (using Homebrew)
```sh
brew update
brew install cmake boost tbb gmp mpfr
```
Once you have installed these packages, you can clone `MINGROC++` onto your local machine using 

```
git clone --recurse-submodules https://github.com/DillonCislo/MINGROCpp.git
```

This will clone the repository and all of the submodules as well. We have also included a script to compile the MATLAB mex files: `MINGROCpp/install_mingroc_matlab.m`. For most Linux/Mac users everything should work smoothly just pressing the big green `Run` button, such that all paths points towards that shipped dependencies in the `MINGROCpp/include/external` folder. Advanced users can customize the installation to point towards non-default dependency locations, if desired. (NOTE: it is probably possible for Windows users to compile all of the mex functions by pointing to the correct dependency installations, but this is not directly supported).
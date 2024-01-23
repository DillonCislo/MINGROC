/*
 * Copyright (C) 2024 Dillon Cislo
 *
 * This file is part of MINGROC++.
 *
 * MINGROC++ is free software: you can redistribute it and/or modify it under the terms
 * of the GNU General Public License as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will by useful, but WITHOUT ANY WARRANTY;
 * without even the implied warranty of MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program.
 * If not, see <http://www.gnu.org/licenses/>
 *
 */

#ifndef _MINGROC_H_
#define _MINGROC_H_

#include "mingrocInline.h"

#include <vector>
#include <complex>
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "MINGROCParam.h"
#include "../external/NNIpp/include/NaturalNeighborInterpolant/NaturalNeighborInterpolant.h"
#include "../external/NNIpp/include/NaturalNeighborInterpolant/NNIParam.h"

namespace MINGROCpp {


  ///
  /// A class to calculate the minimum information constant growth pattern
  /// between two surfaces.
  ///
  /// Templates:
  ///
  ///   Scalar    The input type of data points and function values
  ///   Index     The data type of the triangulation indices
  ///
  template <typename Scalar, typename Index>
  class MINGROC {

    public:

      typedef std::complex<Scalar> CScalar;
      typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
      typedef Eigen::Matrix<Scalar, 1, Eigen::Dynamic> RowVector;
      typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
      typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> ArrayVec;
      typedef Eigen::Array<Scalar, 1, Eigen::Dynamic> ArrayRowVec;
      typedef Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic> Array;
      typedef Eigen::Matrix<CScalar, Eigen::Dynamic, 1> CplxVector;
      typedef Eigen::Matrix<CScalar, 1, Eigen::Dynamic> CplxRowVector;
      typedef Eigen::Matrix<CScalar, Eigen::Dynamic, Eigen::Dynamic> CplxMatrix;
      typedef Eigen::Array<CScalar, Eigen::Dynamic, 1> CplxArrayVec;
      typedef Eigen::Array<CScalar, Eigen::Dynamic, Eigen::Dynamic> CplxArray;
      typedef Eigen::Matrix<Index, Eigen::Dynamic, 1> IndexVector;
      typedef Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic> IndexMatrix;
      typedef Eigen::Array<Index, Eigen::Dynamic, 1> IndexArrayVec;
      typedef Eigen::Array<Index, Eigen::Dynamic, Eigen::Dynamic> IndexArray;

      // Mesh Properties ----------------------------------------------------------------

      // #F by 3 matrix. Discrete surface face connectivity list
      IndexMatrix m_F;

      // #V by 3 matrix. 3D discrete surface vertex coordinate list
      Matrix m_V;

      // #V by 2 matrix. 2D pullback coordinate list.
      // Defines the domain of parameterization
      Matrix m_x;

      // #E by 2 matrix. Discrete surface edge connectivity list
      IndexMatrix m_E;

      // #F by 3 matrix. m_FE(i,j) is the ID of the edge opposite
      // vertex j in face i
      IndexMatrix m_FE;

      // #E by 1 vector of vectors. Contains the IDs of faces attached
      // to unoriented edges
      std::vector<std::vector<Index> > m_EF;

      // #V by #F sparse matrix operator. Averages face-based quantities onto
      // vertices using angle weighting in the domain of parameterization
      Eigen::SparseMatrix<Scalar> m_F2V;

      // #V by #F sparse matrix operator. Averages face-based quantities onto
      // vertices using simple numeric averagin
      Eigen::SparseMatrix<Scalar> m_F2VRaw;

      // #F by #V sparse matrix operator. Averages vertex-based quantities
      // onto faces
      Eigen::SparseMatrix<Scalar> m_V2F;

      // #F by 1 list of face areas in the 2D pullback mesh
      Vector m_AF2D;

      // #V by 1 list of barycentric vertex areas in the 2D pullback mesh
      Vector m_AV2D;

      // The total area of the 2D pullback mesh (should be ~pi)
      Scalar m_A2DTot;

      // #V by #V array. Holds a replicated version of 'm_vertexAreas' used
      // to calculate the update in the quasiconformal mapping
      // Equal to m_vertexAreas.transpose().replicate(#V, 1)
      Array m_AV2DMat;

      // #F by 1 list of face areas in the initial 3D surface
      Vector m_AF3D0;

      // #F by 1 list of double face areas in the initial 3D surface
      Vector m_dblA0_F;

      // #V by 1 list of barycentric vertex areas in the initial 3D surface
      Vector m_AV3D0;

      // The total area of the intial 3D surface
      Scalar m_A3D0Tot;

      // A vector of all vertex IDs on the mesh boundary
      IndexVector m_bdyIDx;

      // A vector of all vertex IDs in the bulk
      IndexVector m_bulkIDx;

      // #2V by #2V sparse identity matrix. Used to calculate abbreviated gradients
      // with respect to fully composed parameterization
      Eigen::SparseMatrix<Scalar> m_speye;

      // Mesh Differential Operators ----------------------------------------------------

      // #F by #V sparse matrix operator df/dx
      Eigen::SparseMatrix<Scalar> m_Dx;

      // #F by #V sparse matrix operator df/dy
      Eigen::SparseMatrix<Scalar> m_Dy;

      // #F by #V sparse matrix operator df/dz
      Eigen::SparseMatrix<CScalar> m_Dz;

      // #F by #V sparse matrix operator df/dz*
      Eigen::SparseMatrix<CScalar> m_Dc;

      // #V by #V Laplace-Beltrami operator in the 2D pullback space
      // NOTE: This matrix uses the convention that L2D is **positive**
      // semi-definite (minus the usual 'libigl' convention)
      Eigen::SparseMatrix<Scalar> m_L2D;

      // #F by #F Laplace-Beltrami operator in the 2D pullback space
      // NOTE: This matrix is just equal to (m_F2V.transpose * m_L2D * mF2V)
      Eigen::SparseMatrix<Scalar> m_LF2D;

      // #V by #V Laplace-Beltrami operator in the initial 3D surface
      // NOTE: This matrix uses the convention that L3D is **positive**
      // semi-definite (minus the usual 'libigl' convention)
      Eigen::SparseMatrix<Scalar> m_L3D0;

      // #F by #F Laplace-Beltrami operator in the initial 3D surface
      // NOTE: This matrix is just equal to (m_F2V.transpose * m_L3D0 * mF2V)
      Eigen::SparseMatrix<Scalar> m_LF3D0;

      // Surface Interpolant Properties -------------------------------------------------

      // Natureal neighbor interpolation parameters
      NNIpp::NNIParam<Scalar> m_nniParam;

      // The natural neighbor interpolant representing the final discrete surface
      NNIpp::NaturalNeighborInterpolant<Scalar> m_NNI;

      // Optimization Properties --------------------------------------------------------

      MINGROCpp::MINGROCParam<Scalar> m_param;

    public:

      ///
      /// Default constructor
      ///
      /// Inputs:
      ///
      ///   F           #F by 3 face connectivity list
      ///   V           #V by 3 initial 3D vertex coordinate list
      ///   x           #V by 2 2D vertex coordinate list
      ///   mingroParam A 'MINGROCParam' class containing the optimization parameters
      ///   nniParam    An 'NNIParam' class containing the parameters needed to
      ///               construct the surface interpolant
      ///
      MINGROC( const IndexMatrix &F, const Matrix &V, const Matrix &x,
          const MINGROCParam<Scalar> &mingroParam,
          const NNIpp::NNIParam<Scalar> &nniParam );

      ///
      /// Calculate the minimum information constant growth pattern for a
      /// given target surface
      ///
      /// Inputs:
      ///
      ///   finMap3D  #V by 3 list of 3D vertex coordinates in the final target
      ///             configuration
      ///
      ///   initMu    #V by 1 initial guess for the Beltrami coefficient
      ///             that specifies the mapping corresponding to the minimum
      ///             distance SEM
      ///
      ///   initMap   #V by 1 complex representation of the quasiconformal
      ///             mapping specified by 'initMu'
      ///
      ///   fixIDx    #P by 1 list of vertex IDs that are to be fixed under the
      ///             quasiconformal mapping
      ///
      ///
      /// Outputs:
      ///
      ///   EG    The minimum information constant growth energy
      ///
      ///   mu    #V by 1 complex representation of the Beltrami coefficient
      ///         that specifies the mapping corresponding to the minimum
      ///         information constant growth pattern
      ///
      ///   w     #V by 1 complex representation of the quasiconformal mapping
      ///         corresponding to the minimum information constant growth pattern
      ///
      ///   map3D #V by 3 list of vertex coordinates in the 3D parameterization
      ///         corresponding to the minimum information constant growth pattern
      ///
      Scalar operator() ( const Matrix &finMap3D, const CplxVector &initMu,
          const CplxVector &initMap, const IndexVector &fixIDx,
          CplxVector &mu, CplxVector &w, Matrix &map3D ) const;

      ///
      /// Calculate the MINGROC energy functional for a given configuration
      ///
      /// Inputs:
      ///
      ///   finMap3D    #V by 3 list of vertex coordinates in the final 3D
      ///               configuration
      ///
      ///   mu      #V by 1 list of complex Beltrami coefficients specifying
      ///           the mapping corresponding to the minimum information
      ///           constant growth pattern
      ///
      ///   w       #V by 1 complex representation of the quasiconformal mapping
      ///           corresponding to the minimum information constant growth
      ///           pattern
      ///
      /// Outputs:
      ///
      ///   E       The total energy for the input configuration
      ///
      ///   map3D   #V by 3 list of vertex coordinates in the updated configuration
      ///
      ///   gamma   #V by 1 list of vertex area ratios in the updated configuration
      ///
      Scalar calculateEnergy (
          const CplxVector &mu, const CplxVector &w,
          Matrix &map3D, Vector &gamma ) const;

      ///
      /// Calculate the MINGROC energy functional and gradients for a given
      /// configuration
      ///
      /// Inputs:
      ///
      ///   finMap3D    #V by 3 list of vertex coordinates in the final 3D
      ///               configuration
      ///
      ///   mu      #V by 1 list of complex Beltrami coefficients specifying
      ///           the mapping corresponding to the minimum distance SEM
      ///
      ///   w       #V by 1 complex representation of the quasiconformal mapping
      ///           corresponding to the minimum distance SEM. NOTE: does NOT
      ///           include the post-composition with a Mobius transformation
      ///
      ///   G1      #V by #V array. Coefficient of nu1 in the real part of K
      ///
      ///   G2      #V by #V array. Coefficient of nu2 in the real part of K
      ///
      ///   G3      #V by #V array. Coefficient of nu1 in the imaginary part of K
      ///
      ///   G4      #V by #V array. Coefficient of nu2 in the imaginary part of K
      ///
      /// Outputs:
      ///
      ///   E           The total energy for the input configuration
      ///
      ///   gradMu      #V by 1 complex gradient vector wrt the Beltrami coefficient
      ///
      ///   map3D   #V by 3 list of vertex coordinates in the updated configuration
      ///
      ///   gamma   #V by 1 list of vertex area ratios in the updated configuration
      ///
      Scalar calculateEnergyAndGrad (
          const CplxVector &mu, const CplxVector &w,
          const Array &G1, const Array &G2, const Array &G3, const Array &G4,
          CplxVector &gradMu, Matrix &map3D, Vector &gamma ) const;

      ///
      /// Construct the components of the kernel that is integrated to find the
      /// variation in the quasiconformal mapping under the variation of the
      /// Beltrami coefficient
      ///
      /// Inputs:
      ///
      ///   w     #V by 1 complex representation of a quasiconformal mapping
      ///
      /// Outputs:
      ///
      ///   G1      #V by #V array. Coefficient of nu1 in the real part of K
      ///
      ///   G2      #V by #V array. Coefficient of nu2 in the real part of K
      ///
      ///   G3      #V by #V array. Coefficient of nu1 in the imaginary part of K
      ///
      ///   G4      #V by #V array. Coefficient of nu2 in the imaginary part of K
      ///
      MINGROC_INLINE void calculateMappingKernel ( const CplxVector &w,
          Array &G1, Array &G2, Array &G3, Array &G4 ) const;

      ///
      /// Calculates the change in the quasiconformal mapping for a given variation
      /// in its associated Beltrami coefficient according to the Beltrami
      /// Holomorphic Flow
      ///
      /// Inputs:
      ///
      ///   nu      #V by 1 complex representation of the variation in the Beltrami
      ///           coefficient
      ///
      ///   G1      #V by #V array. Coefficient of nu1 in the real part of K
      ///
      ///   G2      #V by #V array. Coefficient of nu2 in the real part of K
      ///
      ///   G3      #V by #V array. Coefficient of nu1 in the imaginary part of K
      ///
      ///   G4      #V by #V array. Coefficient of nu2 in the imaginary part of K
      ///
      /// Outputs:
      ///
      ///   dW      #V by 1 change in the mapping
      ///
      MINGROC_INLINE void calculateMappingUpdate ( const CplxVector &nu,
          const Array &G1, const Array &G2,
          const Array &G3, const Array &G4, CplxVector &dW ) const;

      ///
      /// Construct the intrinsic differential operators on the surface mesh
      ///
      void constructDifferentialOperators();

      ///
      /// Construct the mesh function averaging operators
      ///
      void constructAveragingOperators();

      ///
      /// A helper function that can populate the MINGROC properties
      /// for a given input mesh describing the initial condition.
      ///
      /// Inputs:
      ///
      ///   F   #Fx3 face connectivity list
      ///
      ///   V   #Vx3 3D vertex coordinate list in the initial surface
      ///
      ///   x   #Vx2 2D vertex coordinates in the pullback space
      ///
      void buildMINGROCFromMesh( const IndexMatrix &F,
          const Matrix &V, const &x );

      ///
      /// Convert a stacked real vector into complex format
      ///
      /// Inputs:
      ///
      ///   x   (2#N) by 1 real vector
      ///
      /// Outputs:
      ///
      ///   z   #N by 1 complex vector
      ///
      MINGROC_INLINE void convertRealToComplex( const Vector &x, CplxVector &z );

  };

}

#ifndef MINGROC_STATIC_LIBRARY
#  include "MINGROC.cpp"
#endif

#endif


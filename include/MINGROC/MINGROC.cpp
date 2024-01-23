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

#include "MINGROC.h"
#include "clipToUnitCircle.h"
#include "../LBFGS/BFGSMat.h"
#include "../LBFGS/LineSearchBacktracking.h"

#include <cassert>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <omp.h>

#include <igl/cross.h>
#include <igl/edges.h>
#include <igl/internal_angles.h>
#include <igl/boundary_loop.h>
#include <igl/setdiff.h>
#include <igl/edge_lengths.h>
#include <igl/doublearea.h>
#include <igl/unique_edge_map.h>
#include <igl/cotmatrix.h>

// ======================================================================================
// CONSTRUCTOR FUNCTIONS
// ======================================================================================

///
/// Default constructor
///
template <typename Scalar, typename Index>
MINGROCpp::MINGROC<Scalar, Index>::MINGROC(
    const Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic> &F,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &V,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &x,
    const MINGROCParam<Scalar> &mingrocParam,
    const NNIpp::NNIParam<Scalar> &nniParam )
{

  // Check for valid MINGROC parameters
  mingrocParam.checkParam();
  m_param = mingrocParam;

  // Check for valid NNI parameters
  nniParam.checkParam();
  m_nniParam = nniParam;

  // Populate MINGROC properties
  this->buildMINGROCFromMesh(F, V, x);

};

///
/// A helper function that can populate the MINGROC properties
/// for a given input mesh describing the initial condition.
///
template <typename Scalar, typename Index>
void MINGROCpp:MINGROC<Scalar, Index>::buildMINGROCFromMesh(
    const Eigen::Matrix<Index, Eigen::Dynamic, Eigen::Dynamic> &F,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &V,
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &x )
{

  m_F = f;
  m_V = V;
  m_x = x;

  // Verify input triangulation entries
  if ( ( m_F.array().isNaN().any() ) || ( m_F.array().isInf().any() ) )
    std::runtime_error("Invalid face connectivity list");

  if ( ( m_V.array().isNaN().any() ) || ( m_V.array().isInf().any() ) )
    std::runtime_error("Invalid 3D vertex coordinate list");

  if ( ( m_x.array().isNaN().any() ) || ( m_x.array().isInf().any() ) )
    std::runtime_error("Invalid 2D vertex coordinate list");

  if ( m_V.rows() != m_x.rows() )
    std::runtime_error("Inconsistent number of vertices");

  // Construct the triangulation edge list
  IndexMatrix dE(1,2); // Directed edge list
  m_E = IndexMatrix::Zero(1,2); // Undirected edge list
  IndexVector FE(1);
  std::vector<std::vector<Index> > uE2E; // Unused - shouldn't be necessary...
  igl::unique_edge_map( m_F, dE, m_E, FE, uE2E );

  // Reserve space for the edge-face correspondence list
  m_EF.reserve( m_E.rows() );
  for( int i = 0; i < m_E.rows(); i++ )
  {
    std::vector<Index> tmpVec;
    m_EF.push_back(tmpVec);
  }

  // Assemble the face-edge and edge-face correspondence lists
  m_FE.resizeLike( m_F );
  for( int i = 0; i < m_F.rows(); i++ )
  {
    for( int j = 0; j < 3; j++ )
    {
      m_FE(i,j) = FE( i + (j * m_F.rows()) );
      m_EF[ m_FE(i,j) ].push_back( (Index) i );
    }
  }

  // Verify that the triangulation is a topological disk
  int eulerChi = m_x.rows() - m_E.rows() + m_F.rows();
  if ( eulerChi != 1 )
    std::runtime_error("Input mesh must be a topological disk");

  // Extract bulk/boundary vertices
  std::vector<std::vector<Index> > allBdyLoops;
  igl::boundary_loop( m_F, allBdyLoops );

  if ( allBdyLoops.size() != 1 )
    std::runtime_error("Input mesh has more than one boundary");

  m_bdyIDx = IndexVector::Zero(allBdyLoops[0].size(), 1);
  igl::boundary_loop( m_F, m_bdyIDx );

  IndexVector allVIDx = IndexVector::LinSpaced(m_V.rows(), 0, (m_V.rows()-1));
  IndexVector IA(1, 1); // Not used - try to find way to eliminate
  m_bulkIDx = IndexVector::Zero(1,1);

  igl::setdiff( allVIDx, m_bdyIDx, m_bulkIDx, IA );

  // Clip the boundary vertices to the unit circle
  MINGROCpp::clipToUnitCircle( m_bdyIDx, m_x );

  // Check that no points in the pullback mesh lie outside the unit disk
  if ( ( m_x.rowwise().norm().array() > Scalar(1.0) ).any() )
    std::runtime_error("Some pullback vertices lie outside the unit disk");

  // Construct averaging operators
  this->constructAveragingOperators();

  // Construct the differential operators
  // NOTE: THIS MUST COME AFTER CONSTRUCTING THE AVERAGING OPERATORS
  this->constructDifferentialOperators();

  // Compute the areas of each face in the initial 3D mesh
  m_dblA0_F = Vector::Zero(m_F.rows(), 1);
  igl::doublearea( m_V, m_F, m_dblA0_F );
  m_AF3D0 = (m_dblA0_F.array() / Scalar(2.0)).matrix();
  m_A3D0Tot = m_AF3D0.array().sum();

  // Compute the areas of each face in the 2D pullback mesh
  m_AF2D = Vector::Zero(m_F.rows(), 1);
  igl::doublearea( m_x, m_F, m_AF2D );
  m_AF2D = (m_AF2D.array() / Scalar(2.0)).matrix();
  m_A2DTot = m_AF2D.array().sum();

  // Compute vertex areas from face areas
  m_AV3D0 = Vector::Zero(m_V.rows(), 1);
  m_AV2D = Vector::Zero(m_V.rows(), 1);
  for( int i = 0; i < m_F.rows(); i++ )
  {
    for( int j = 0; j < m_V.rows(); j++ )
    {
      m_AV3D0(m_F(i,j)) += m_AF3D0(i);
      m_AF2D(m_F(i,j)) += m_AF2D(i);
    }
  }

  m_AV3D0 = (m_AV3D0.array() / Scalar(3.0)).matrix();
  m_AV2D = (m_AV2D.array() / Scalar(3.0)).matrix();

  // Construct the vertex area matrix
  RowVector VAT = m_AV2D.transpose();
  m_AV2DMat = VAT.replicate( m_x.rows(), 1);

  // Construct the sparse identity matrix
  typedef Eigen::Triplet<Scalar> T;
  std::vector<T> tListI;
  tListI.reserve( 2 * m_V.rows() );
  for( int i = 0; i < (2 * m_V.rows()); i++ )
    tListI.push_back( T(i, i, Scalar(1.0)) );

  Eigen::SparseMatrix<Scalar> speye( 2 * m_V.rows(), 2 * m_V.rows() );
  speye.setFromTriplets( tListI.begin(), tListI.end() );
  m_speye = speye;

};

///
/// Construct the intrinsic differential operators on the surface mesh
///
template <typename Scalar, typename Index>
void MINGROCpp::MINGROC<Scalar, Index>::constructDifferentialOperators() {

  // The number of vertices
  int numV = m_x.rows();

  // The number of faces
  int numF = m_F.rows();

  // A vector of face indices
  IndexVector u(numF, 1);
  u = IndexVector::LinSpaced(numF, 0, (numF-1));

  // Extract facet edges
  // NOTE: This expects that the faces are oriented CCW
  Eigen::Matrix<Scalar, Eigen::Dynamic, 2> e1(numF, 2);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 2> e2(numF, 2);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 2> e3(numF, 2);
  for( int i = 0; i < numF; i++ ) {

    e1.row(i) = m_x.row( m_F(i,2) ) - m_x.row( m_F(i,1) );
    e2.row(i) = m_x.row( m_F(i,0) ) - m_x.row( m_F(i,2) );
    e3.row(i) = m_x.row( m_F(i,1) ) - m_x.row( m_F(i,0) );

  }

  Eigen::Matrix<Scalar, Eigen::Dynamic, 3> eX(numF, 3);
  Eigen::Matrix<Scalar, Eigen::Dynamic, 3> eY(numF, 3);
  eX << e1.col(0), e2.col(0), e3.col(0);
  eY << e1.col(1), e2.col(1), e3.col(1);

  // Extract signed facet areas
  Vector a = ( e1.col(0).cwiseProduct( e2.col(1) )
      - e2.col(0).cwiseProduct( e1.col(1) ) ) / Scalar(2.0);

  // Construct sparse matrix triplets
  typedef Eigen::Triplet<Scalar> T;
  typedef Eigen::Triplet<CScalar> TC;

  std::vector<T> tListX, tListY;
  std::vector<TC> tListZ, tListC;

  tListX.reserve( 3 * numF );
  tListY.reserve( 3 * numF );
  tListZ.reserve( 3 * numF );
  tListC.reserve( 3 * numF );

  for( int i = 0; i < numF; i++ ) {
    for( int j = 0; j < 3; j++ ) {

      Scalar mx = -eY(i,j) / ( Scalar(2.0) * a(i) );
      Scalar my = eX(i,j) / ( Scalar(2.0) * a(i) );
      CScalar mz( mx / Scalar(2.0), -my / Scalar(2.0) );
      CScalar mc( mx / Scalar(2.0), my / Scalar(2.0) );

      tListX.push_back( T( u(i), m_F(i,j), mx ) );
      tListY.push_back( T( u(i), m_F(i,j), my ) );
      tListZ.push_back( TC( u(i), m_F(i,j), mz ) );
      tListC.push_back( TC( u(i), m_F(i,j), mc ) );

    }
  }

  // Complete sparse operator construction
  Eigen::SparseMatrix<Scalar> Dx( numF, numV );
  Eigen::SparseMatrix<Scalar> Dy( numF, numV );
  Eigen::SparseMatrix<CScalar> Dz( numF, numV );
  Eigen::SparseMatrix<CScalar> Dc( numF, numV );

  Dx.setFromTriplets( tListX.begin(), tListX.end() );
  Dy.setFromTriplets( tListY.begin(), tListY.end() );
  Dz.setFromTriplets( tListZ.begin(), tListZ.end() );
  Dc.setFromTriplets( tListC.begin(), tListC.end() );

  // Set member variables
  m_Dx = Dx;
  m_Dy = Dy;
  m_Dz = Dz;
  m_Dc = Dc;

  // Construct Laplace-Beltrami operator on the 2D pullback space
  Eigen::SparseMatrix<Scalar> L2D( numV, numV );
  igl::cotmatrix(m_x, m_F, L2D);
  L2D = Scalar(-1.0) * L2D;
  m_L2D = L2D;
  m_LF2D = m_F2V.transpose() * L2D * m_F2V;


  // Construct Laplace-Beltrami operator on the 3D initial surface
  Eigen::SparseMatrix<Scalar> L3D0( numV, numV );
  igl::cotmatrix(m_V, m_F, L3D0);
  L3D0 = Scalar(-1.0) * L3D0;
  m_L3D0 = L3D0;
  m_LF3D0 = m_F2V.transpose() * L3D0 * m_F2V;

};

///
/// Construct the mesh function averaging operators
///
template <typename Scalar, typename Index>
void MINGROCpp::MINGROC<Scalar, Index>::constructAveragingOperators() {

  typedef Eigen::Triplet<Scalar> T;

  // The number of vertices
  int numV = m_x.rows();

  // The number of faces
  int numF = m_F.rows();

  // A vector of face indices
  IndexVector fIDx(numF, 1);
  fIDx = IndexVector::LinSpaced(numF, 0, (numF-1));

  // Extract the internal angles of the 2D triangulation
  Matrix fangles(numF, 3);
  igl::internal_angles( m_x, m_F, fangles );

  // Normalize the sums of internal angles around each vertex
  std::vector<std::vector<Scalar>> vangles;
  vangles.reserve( numV );

  std::vector<std::vector<Index>> vfIDx;
  vfIDx.reserve( numV );

  Vector vangleSum = Vector::Zero( numV, 1 );
  for( int i = 0; i < numV; i++ )
  {
    std::vector<Scalar> tmpScalarVec;
    vangles.push_back( tmpScalarVec );

    std::vector<Index> tmpIndexVec;
    vfIDx.push_back( tmpIndexVec );
  }

  for( int i = 0; i < numF; i++ )
  {
    for( int j = 0; j < 3; j++ )
    {
      vangles[ m_F(i,j) ].push_back( fangles(i,j) );
      vfIDx[ m_F(i,j) ].push_back( (Index) i );
      vangleSum( m_F(i,j) ) += fangles(i,j);
    }
  }

  for( int i = 0; i < numV; i++ )
    for( int j = 0; j < vangles[i].size(); j++ )
      vangles[i][j] = vangles[i][j] / vangleSum(i);

  // Construct the vertex-to-face averaging operator
  std::vector<T> tListV2F;
  tListV2F.reserve( 3 * numF );

  for( int i = 0; i < numF; i++ )
    for( int j = 0; j < 3; j++ )
      tListV2F.push_back( T( fIDx(i), m_F(i,j), Scalar(1.0/3.0) ) );

  Eigen::SparseMatrix<Scalar> V2F( numF, numV );
  V2F.setFromTriplets( tListV2F.begin(), tListV2F.end() );

  m_V2F = V2F;

  // Construct the face-to-vertex averaging operators
  std::vector<T> tListF2V;
  std::vector<T> tlistF2VRaw;
  tListF2V.reserve( 6 * numV ); // A rough estimate
  tListF2VRaw.reserve( 6 * numV );

  for( int i = 0; i < numV; i++ )
  {
    for( int j = 0; j < vangles[i].size(); j++ )
    {
      tListF2V.push_back( T( (Index) i, vfIDx[i][j], vangles[i][j] ) );
      tListF2VRaw.push_back( T( (Index) i, vfIDx[i][j], Scalar(1.0 / vangles[i].size()) ) );
    }
  }

  Eigen::SparseMatrix<Scalar> F2V( numV, numF );
  F2V.setFromTriplets( tListF2V.begin(), tListF2V.end() );
  m_F2V = F2V;

  Eigen::SparseMatrix<Scalar> F2VRaw( numV, numF );
  F2VRaw.setFromTriplets( tListF2VRaw.begin(), tListF2VRaw.end() );
  m_F2VRaw = F2VRaw;

};

// ======================================================================================
// MAPPING KERNEL FUNCTIONS
// ======================================================================================

///
/// Construct the components of the kernel that is integrated to find the
/// variation in the quasiconformal mapping under the variation of the
/// Beltrami coefficient
///
template <typename Scalar, typename Index>
MINGROC_INLINE void MINGROCpp::MINGROC<Scalar, Index>::calculateMappingKernel (
    const CplxVector &w, Array &G1, Array &G2, Array &G3, Array &G4 ) const {

  // Number of vertices
  int numV = m_V.rows();

  // The squared partial derivative dw/dz defined on vertices
  CplxVector DWz2(numV, 1);
  DWz2.noalias() = m_F2V * m_Dz * w;
  DWz2 = (DWz2.array() * DWz2.array()).matrix();

  // Cached variables for conjugates
  CplxVector wC = w.conjugate();
  CplxVector DWz2C = DWz2.conjugate();

  // More cached variables to avoid duplicate computations
  CplxVector xi = w.array() * (Scalar(1.0) - w.array());
  CplxVector xiC = xi.conjugate();

  CplxVector pixi = Scalar(M_PI) * xi;
  CplxVector pixiC = pixi.conjugate();

  // Construct Complex Coefficients -----------------------------------------------------
  
  #pragma omp parallel for collapse(2)
  for( int i = 0; i < numV; i++ ) {
    for( int j = 0; j < numV; j++ ) {

      CScalar Anum = xi(i) * DWz2(j);
      CScalar Adenom = pixi(j) * (w(i) - w(j));

      if ( std::abs(Adenom) < Scalar(1e-10) ) {
        Adenom = CScalar(Scalar(0.0), Scalar(0.0));
      }

      CScalar Bnum = xi(i) * DWz2C(j);
      CScalar Bdenom = pixiC(j) * (Scalar(1.0) - wC(j) * w(i));

      if ( std::abs(Bdenom) < Scalar(1e-10) ) {
        Bdenom = CScalar(Scalar(0.0), Scalar(0.0));
      }

      CScalar A = Anum / Adenom; // The complex coefficient of nu
      CScalar B = Bnum / Bdenom; // The complex coefficient of nuC

      // Construct real mapping kernel coefficients
      G1(i,j) = A.real() + B.real();
      G2(i,j) = B.imag() - A.imag();
      G3(i,j) = A.imag() + B.imag();
      G4(i,j) = A.real() - B.real();

      // Deal with singularities
      // NOTE: 'isinfinite' is not a templated function
      // One workaround is to convert all input arguments to doubles
      if ( !std::isfinite( (double) G1(i,j)) ) { G1(i,j) = Scalar(0.0); }
      if ( !std::isfinite( (double) G2(i,j)) ) { G2(i,j) = Scalar(0.0); }
      if ( !std::isfinite( (double) G3(i,j)) ) { G3(i,j) = Scalar(0.0); }
      if ( !std::isfinite( (double) G4(i,j)) ) { G4(i,j) = Scalar(0.0); }

    }
  }

};

///
/// Calculates the change in the quasi conformal mapping for a given variation
/// in its associate Beltrami coefficient according to the Beltrami
/// Holomorphic Flow
///
template <typename Scalar, typename Index>
MINGROC_INLINE void MINGROCpp::MINGROC<Scalar, Index>::calculateMappingUpdate (
    const CplxVector &nu, const Array &G1, const Array &G2,
    const Array &G3, const Array &G4, CplxVector &dW ) const {

  // Number of vertices
  int numV = m_V.rows();

  // Cannot transpose in place with Eigen
  CplxRowVector nuT = nu.transpose();

  // The real part of the variation
  RowVector nu1 = nuT.real();
  Array nu1Arr = nu1.replicate( numV, 1 );

  // The imaginary part of the variation
  RowVector nu2 = nuT.imag();
  Array nu2Arr = nu2.replicate( numV, 1 );

  // Calculate the update step
  CplxArray dWArr(numV, numV);
  dWArr.real() = G1 * nu1Arr + G2 * nu2Arr;
  dWArr.imag() = G3 * nu1Arr + G4 * nu2Arr;
  dWArr = dWArr * m_AV2DMat;

  dW = dWArr.rowwise().sum();

};

// ======================================================================================
// ENERGY AND ENERGY GRADIENT FUNCTIONS
// ======================================================================================

///
/// Calculate the MINGROC energy functional for a given configuration
///
template <typename Scalar, typename Index>
Scalar MINGROCpp::MINGROC<Scalar, Index>::calculateEnergy (
    const CplxVector &mu, const CplxVector &w,
    Matrix &map3D, Vector &gamma ) const
{

  // The number of vertices
  int numV = m_V.rows();

  //-------------------------------------------------------------------------------------
  // Calculate MINGRO Energy
  //-------------------------------------------------------------------------------------
  
  // Calculate updated 3D vertex locations
  map3D = Matrix::Zero(numV, 3);
  m_NNI( w.real(), w.imag(), map3D );

  // Face areas in the updated configuration
  Vector dblA_F(numV);
  igl::doublearea(map3D, m_F, dblA_F);
  
  // Calculate growth factor on vertices
  gamma = m_F2V * ( dblA_F.array() / m_dblA0_F.array() );
  // gamma = m_F2VRaw * ( dblA_F.array() / m_dblA0_F.array() );
  
  // The growth energy
  Scalar E = (gamma.transpose() * m_L2D * gamma) / m_A2DTot;
  // Scalar E = (gamma.transpose() * m_L3D0 * gamma) / m_A3DTot;
  
  //-------------------------------------------------------------------------------------
  // Calculate Conformal Deviation Energy
  //-------------------------------------------------------------------------------------
  
  if ( m_CC > Scalar(0.0) )
  {
    ArrayVec absMu2 = (mu.array() * mu.array().conjugate()).real();
    Scalar ECC = (absMu2.array() * m_AV2D.array()).sum() / m_A2DTot;
    // Scalar ECC = (absMu2.array() * m_AV3D0.array()).sum() / m_A3D0Tot;

    E += m_param.CC * ECC;
  }

  //-------------------------------------------------------------------------------------
  // Calculate Quasiconformal Smoothness Energy
  //-------------------------------------------------------------------------------------
  
  if ( m_SC > Scalar(0.0) )
  {
    ArrayVec muR = mu.array().real();
    ArrayVec muI = mu.array().imag();

    Scalar ESC = (muR.transpose() * m_L2D * muR) + (muI.transpose() * m_L2D * muI);
    ESC = ESC / m_A2DTot;

    // Scalar ESC = (muR.transpose() * m_L3D0 * muR) + (muI.transpose() * m_L3D0 * muI);
    // ESC = ESC / m_A3D0Tot;

    E += m_param.SC * ESC;
  }

  //-------------------------------------------------------------------------------------
  // Calculate Bound Constraint Energy on the Beltrami Coefficient
  //-------------------------------------------------------------------------------------
  
  if ( m_DC > Scalar(0.0) )
  {
    Scalar EDC = ( Scalar(1.0) - mu.array().abs() ).log().sum();
    E -= EDC / m_param.DC;
  }

  return E;

};

///
/// Calculate the MINGROC energy functional and gradients for a given
/// configuration
///
template <typename Scalar, typename Index>
Scalar MINGROCpp::MINGROC<Scalar, Index>::calculateEnergyAndGrad (
    const CplxVector &mu, const CplxVector &w,
    const Array &G1, const Array &G2, const Array &G3, const Array &G4,
    CplxVector &gradMu, Matrix &map3D, Vector &gamma ) const
{

  // The number of vertices
  int numV = m_V.rows();

  // The number of faces
  int numF = m_F.rows();

  typedef Eigen::Triplet<Scalar> T;

  //-------------------------------------------------------------------------------------
  // Calculate MINGRO Energy
  //-------------------------------------------------------------------------------------
  
  // Calculate updated 3D vertex locations
  map3D = Matrix::Zero(numV, 3);
  Matrix Dmap3DDu(numV, 3);
  Matrix Dmap3DDv(numV, 3);
  m_NNI( w.real(), w.imag(), map3D, Dmap3DDu, Dmap3DDv );

  // Face-based edge vectors in the updated configuration
  Matrix ei(numF, 3);
  Matrix ej(numF, 3);
  Matrix ek(numF, 3);
  for( int i = 0; i < numF; i++ )
  {
    ei.row(i) = map3D.row(m_F(i,2)) - map3D.row(m_F(i,1));
    ej.row(i) = map3D.row(m_F(i,0)) - map3D.row(m_F(i,2));
    ek.row(i) = map3D.row(m_F(i,1)) - map3D.row(m_F(i,0));
  }

  // Compute face unit formals and double areas in the updated configuration
  Matrix n(numF, 3);
  igl::cross(ei, ej, n);
  ArrayVec dblA_F = ((n.array() * n.array()).rowwise().sum()).sqrt();
  n = (n.array() / dblA_F.replicate(1, 3)).matrix();

  // Rotated edge vectors in the updated configuration
  Matrix ti(numF, 3); igl::cross(ei, n, ti);
  Matrix tj(numF, 3); igl::cross(ej, n, tj);
  Matrix tk(numF, 3); igl::cross(ek, n, tk);

  // Calculate growth factor on vertices
  Vector gammaF = dblA_F.array() / m_dblA0_F.array();
  gamma = m_F2V * gammaF;
  // gamma = m_F2VRaw * gammaF;
  
  // The growth energy
  Scalar E = (gamma.transpose() * m_L2D * gamma) / m_A2DTot;
  // Scalar E = (gamma.transpose() * m_L3D0 * gamma) / m_A3DTot;
  
  // ------------------------------------------------------------------------------------
  // Calculate Gradients With Respect to the Quasiconformal Parameterization
  // ------------------------------------------------------------------------------------
  
  // Calculate the gradient of the updated double face areas with respect to the
  // quasiconformal parameterization. Entry DgammaFDu(f,i) is the gradient of the
  // face area ratio of face f with respect to the real 2D coordinate of the ith
  // vertex in face f
  Matrix DgammaFDu = Matrix::Zero(numF, 3);
  for( int i = 0; i < numF; i++ )
  {
    DgammaFDu(i,0) = -ti.row(i).dot(Dmap3DDu.row(m_F(i,0)));
    DgammaFDu(i,1) = -tj.row(i).dot(Dmap3DDu.row(m_F(i,1)));
    DgammaFDu(i,2) = -tk.row(i).dot(Dmap3DDu.row(m_F(i,2)));
  }

  DgammaFDu = (DgammaFDu.array() / m_dblA0_F.array().replicate(1,3)).matrix();

  // Calculate the gradient of the updated double face areas with respect to the
  // quasiconformal parameterization. Entry DgammaFDv(f,i) is the gradient of the
  // face area ratio of face f with respect to the imaginary 2D coordinate of the
  // ith vertex in face f
  Matrix DgammaFDv = Matrix::Zero(numF, 3);
  for( int i = 0; i < numF; i++ )
  {
    DgammaFDv(i,0) = -ti.row(i).dot(Dmap3DDv.row(m_F(i,0)));
    DgammaFDv(i,1) = -tj.row(i).dot(Dmap3DDv.row(m_F(i,1)));
    DgammaFDv(i,2) = -tk.row(i).dot(Dmap3DDv.row(m_F(i,2)));
  }

  DgammaFDv = (DgammaFDv.array() / m_dblA0_F.array().replicate(1,3)).matrix();

  // Convert to sparse operators to simplify the vertex gradient calculation.
  // Each row corresponds to a particular face and will have only 3 nonzero
  // entries. Each column represents the derivative of all face area ratios
  // with respect to a particular vertex. Only rows corresponding to faces
  // attached to that particular vertex will have nonzero entries
  std::vector<T> tListDgFDu; tlistDgFDu.reserve(3 * numF);
  std::vector<T> tListDgFDv; tlistDgFDv.reserve(3 * numF);
  for(int i = 0; i < numF; i++ )
  {
    for(int j = 0; j < 3; j++ )
    {
      tListDgFDu.push_back( T(i, m_F(i,j), DgammaFDv(i,j)) );
      tListDgFDv.push_back( T(i, m_F(i,j), DgammaFDv(i,j)) );
    }
  }

  Eigen::SparseMatrix<Scalar> DgammaFDu_Mat(numF, numV);
  DgammaFDu_Mat.setFromTriplets( tListDgFDu.begin(), tListDgFDu.end() );

  Eigen::SparseMatrix<Scalar> DgammaFDv_Mat(numF, numV);
  DgammaFDv_Mat.setFromTriplets( tListDgFDv.begin(), tListDgFDv.end() );

  ArrayVec dEdu = (gammaF.transpose() * m_LF2D * DgammaFDu_Mat).transpose().array();
  dEdu = Scalar(2.0) * dEdu / m_A2DTot;

  ArrayVec dEdv = (gammaF.transpose() * m_LF2D * DgammaFDv_Mat).transpose().array();
  dEdv = Scalar(2.0) * dEdv / m_A2DTot;

  /*
  ArrayVec dEdu = (gammaF.transpose() * m_LF3D0 * DgammaFDu_Mat).transpose().array();
  dEdu = Scalar(2.0) * dEdu / m_A3D0Tot;

  ArrayVec dEdv = (gammaF.transpose() * m_LF3D0 * DgammaFDv_Mat).transpose().array();
  dEdv = Scalar(2.0) * dEdv / m_A3D0Tot;
  */

  // ------------------------------------------------------------------------------------
  // Calculate Gradients With Respect to the Beltrami Coefficient
  // ------------------------------------------------------------------------------------
  
  gradMu = CplxVector::Zero(numV);

  #pragma omp parallel for if (numV > 500)
  for( int i = 0; i < numV; i++ ) {


    /* I'm including this bit of inefficient code as a comment
     * to make the structure of the chain rule derivatives for mu
     * more transparent

    // Each of these is a #V by 1 matrix
    ArrayVec dudmu1 = G1.col(i);
    ArrayVec dudmu2 = G2.col(i);
    ArrayVec dvdmu1 = G3.col(i);
    ArrayVec dvdmu2 = G4.col(i);

    Scalar dEdmu1 = (dEdu * dudmu1 + dEdv * dvdmu1).sum();
    Scalar dEdmu2 = (dEdu * dudmu2 + dEdv * dvdmu2).sum();

    */

    Scalar dEdmu1 = (dEdu * G1.col(i) + dEdv * G3.col(i)).sum();
    Scalar dEdmu2 = (dEdu * G2.col(i) + dEdv * G4.col(i)).sum();

    gradMu(i) = CScalar(dEdmu1, dEdmu2);

  }
  
  //-------------------------------------------------------------------------------------
  // Calculate Conformal Deviation Energy
  //-------------------------------------------------------------------------------------
  
  if ( m_CC > Scalar(0.0) )
  {
    ArrayVec absMu2 = (mu.array() * mu.array().conjugate()).real();

    Scalar ECC = (absMu2.array() * m_AV2D.array()).sum() / m_A2DTot;
    CplxArrayVec gradMuCC = Scalar(2.0) * mu.array() * m_AV2D.array() / m_A2DTot;

    /*
    Scalar ECC = (absMu2.array() * m_AV3D0.array()).sum() / m_A3D0Tot;
    CplxArrayVec gradMuCC = Scalar(2.0) * mu.array() * m_AV3D0.array() / m_A3D0Tot;
    */

    E += m_param.CC * ECC;
    gradMu = (gradMu.array() + m_param.CC * gradMuCC).matrix();
  }

  //-------------------------------------------------------------------------------------
  // Calculate Quasiconformal Smoothness Energy
  //-------------------------------------------------------------------------------------
  
  if ( m_SC > Scalar(0.0) )
  {
    ArrayVec muR = mu.array().real();
    ArrayVec muI = mu.array().imag();

    ArrayVec LmuR = (m_L2D * muR).array();
    ArrayVec LmuI = (m_L2D * muI).array();
    Scalar ESC = ((muR.transpose() * LmuR) + (muI.transpose() * LmuI)) / m_A2DTot;
    CplxArrayVec gradMuSC = Scalar(2.0) * (LmuR + CScalar(0.0, 1.0) * LmuI) / m_A2DTot;

    /*
    ArrayVec LmuR = (m_L3D0 * muR).array();
    ArrayVec LmuI = (m_L3D0 * muI).array();
    Scalar ESC = ((muR.transpose() * LmuR) + (muI.transpose() * LmuI)) / m_A3D0Tot;
    CplxArrayVec gradMuSC = Scalar(2.0) * (LmuR + CScalar(0.0, 1.0) * LmuI) / m_A3D0Tot;
    */
    
    E += m_param.SC * ESC;
    gradMu = (gradMu.array() + m_param.SC * gradMuSC).matrix();
  }

  //-------------------------------------------------------------------------------------
  // Calculate Bound Constraint Energy on the Beltrami Coefficient
  //-------------------------------------------------------------------------------------
  
  if ( m_DC > Scalar(0.0) )
  {
    Scalar EDC = ( Scalar(1.0) - mu.array().abs() ).log().sum();
    E -= EDC / m_param.DC;

    ArrayVec absMu = mu.array().abs();
    gradMu = ( gradMu.array() +
        mu.array() / ( absMu * (Scalar(1.0) - absMu) ) / m_param.DC ).matrix();
  }

  return E;

};

// ======================================================================================
// BELTRAMI HOLOMORPHIC FLOW
// ======================================================================================

///
/// Convert a stacked real vector into complex format
///
template <typename Scalar, typename Index>
MINGROC_INLINE void MINGROCpp::MINGROC<Scalar, Index>::convertRealToComplex(
    const Vector &x, CplxVector &z )
{

  z.resize(x.size());
  for( int i = 0; i < x.size(); i++ )
    z(i) = CScalar(x(i), x(i+x.size()));

};

///
/// Calculate the minimum information constant growth pattern for a
/// given target surface
template <typename Scalar, typename Index>
Scalar MINGROCpp::MINGROC<Scalar, Index>::operator() (
    const Matrix &finMap3D, const CplxVector &initMu,
    const CplxVector &initMap, const IndexVector &fixIDx,
    CplxVector &mu, CplxVector &w, Matrix &map3D ) const
{
  
  int numV = m_V.rows(); // Number of vertices 
  int numF = m_F.rows(); // Number of faces
  int numE = m_E.rows(); // Number of edges

  //-------------------------------------------------------------------------------------
  // Input Processing
  //-------------------------------------------------------------------------------------
  
  // Check final surface coordinate list size
  if ((finMap3D.rows() != numV) || (finMap3D.cols() != 3))
    throw std::runtime_error("Improperly sized final 3D surface coordinates");

  // Check final surface coordinate list entries
  for( int i = 0; i < numV; i++ )
    for( int j = 0; j < 3; j++ )
      if ( !std::isfinite( (double) finMap3D(i,j) ) )
        throw std::runtime_error("Non-finite final 3D surface coordinates");

  // Initial guess processing -----------------------------------------------------------
 
  // Initial guess for final embedding is just the input 3D triangulation
  map3D = finMap3D;
  
  // Set the initial 2D quasiconformal parameterization from input
  w = initMap;
  if (w.size() != numV)
    throw std::runtime_error("Initial map is improperly sized");

  // Clip the boundary vertices of w to the unit circle
  MINGROCpp::clipToUnitCircle( m_bdyIDx, w );

  if (m_param.recomputeMu)
  {

    // Just recompute the mu directly from the map
    mu = m_F2V * ((m_Dc * w).array() / (m_Dz * w).array()).matrix();

  } else {

    if (initMu.size() == numV)
      mu = initMu;
    else if ( initMu.size() == numF )
      mu = m_F2V * initMu;
    else
      throw std::runtime_error("Input Beltrami coefficient is improperly sized");

  }

  if ( (mu.array().abs() >= 1.0).any() ) {
    throw std::invalid_argument("Invalid initial Beltrami coefficient");
  }

  if ( (w.array().abs() > 1.0).any() ) {
    throw std::invalid_argument("Invalid initial quasiconformal mapping");
  }

  // Generate the final surface interpolant
  try
  {

    m_NNI = NNIpp::NaturalNeighborInterpolant<Scalar>(
        w.real(), w.imag(), finMap3D, nniParam);

  } catch ( const std::runtime_error &ere ) { throw; }
  } catch ( const std::logic_error &ele ) { throw; }
  } catch ( const std::invalid_argument &eie ) { throw; }

  // Fixed point processing -------------------------------------------------------------
  
  if ( fixedIDx.size() > 0 )
    if ( !((fixedIDx.array() >= Index(0)).all() && (fixedIDx.array() < numV).all()) )
      throw std::runtime_error("User supplied fixed point is out of bounds");

  // ------------------------------------------------------------------------------------
  // Run the Beltrami Holomorphic Flow via L-BFGS
  // ------------------------------------------------------------------------------------
  
  // Optimization Pre-Processing --------------------------------------------------------
  
  // The objective function value
  Scalar fx;

  // The immediately previous function value
  Scalar fx_prev;
  
  // Approximation to the Hessian matrix
  MINGROCpp::BFGSMat<Scalar> bfgs;
  bfgs.reset(2 * numV, m_param.m);

  // Current Beltrami coefficient (real format: [muR; muI])
  Vector x(2 * numV, 1);

  // Old Beltrami coefficient (real format)
  Vector xp(2 * numV, 1);

  // Old quasiconformal map
  CplxVector wp(numV, 1);

  // Old Beltrami coefficient (complex format)
  CplxVector mup(numV, 1);

  // Old 3D embedding
  Matrix map3Dp(numV, 3);

  // New gradient wrt the Beltrami coefficient (real format)
  Vector grad(2 * numV, 1);
  
  // Old gradient wrt the Beltrami coefficient (real format)
  Vector gradp(2 * numV, 1);

  // The update direction for the Beltrami coefficient (real format)
  Vector drt(2 * numV, 1);

  // The update direction for the Beltrami coefficient (complex format)
  CplxVector dmu(numV, 1);

  // The update direction for the quasiconformal mapping
  CplxVector dw(numV, 1);

  // The length of the lag for objective function values to test convergence
  const int fpast = m_param.past;
  
  // History of objective function values
  Vector fxp;
  if ( fpast > 0 ) { fxp.resize(fpast); }

  // Handle the '0th' Iteration ---------------------------------------------------------
  
  // Calculate the mapping kernel
  Array G1( numV, numV );
  Array G2( numV, numV );
  Array G3( numV, numV );
  Array G4( numV, numV );
  calculateMappingKernel( w, G1, G2, G3, G4 );

  // Evaluate the function and gradient for initial configuration
  CplxVector gradMu(numV, 1);
  Vector gamma(numV, 1);
  fx = calculateEnergyAndGrad(mu, w, G1, G2, G3, G4, gradMu, map3D, gamma);
  
  // Convert complex state format -> real state format
  x << mu.real(), mu.imag();
  grad << gradMu.real(), gradMu.imag();

  // Updated vector norms
  Scalar xnorm = x.norm();
  Scalar gnorm = grad.norm();

  if ( m_param.iterDisp )
  {
    std::cout << "(0)"
      << " f(x) = " << std::setw(15) << std::setprecision(10) << fx
      << " ||x|| = " << std::setprecision(4) << std::setw(6) << xnorm
      << " ||dx|| = " << std::setw(10) << std::setprecision(7) << gnorm
      << " |df| = NaN";
  }

  if ( fpast > 0 ) { fxp[0] = fx; }

  // Handle NaNs produced by the initial guess
  if ( (fx != fx) || (xnorm != xnorm) || (gnorm != gnorm) ) {
      throw std::invalid_argument("Initial guess generates NaNs");
  }

  // Handle Infs produced by the initial guess
  if ( std::isinf((double) fx) || std::isinf((double) xnorm) || std::isinf((double) gnorm )) {
    throw std::invalid_argument("Initial guess generates Infs");
  }

  // Early exit if the initial guess is already a minimizer
  if ( gnorm <= m_param.epsilon || gnorm <= m_param.epsilonRel * xnorm ) {
    return fx;
  }

  // Initial update direction
  drt.noalias() = -grad;

  // Convert real state format -> complex state format
  convertRealToComplex(drt, dmu)

  // Calculate the inital update direction for the quasiconformal mapping
  calculateMappingUpdate(dmu, G1, G2, G3, G4, dw);

  // Initial step size
  Scalar step = Scalar(1.0) / drt.norm();

  // Handle All Subsequent Iterations ---------------------------------------------------
  
  int iterNum = 0;
  for( ; ; ) {

    // Increment iteration number
    iterNum++;

    // CONVERGENCE TEST -- Maximum number of iterations
    if ( m_param.maxIterations != 0 && iterNum > m_param.maxIterations )
    {
      if (m_param.iterDisp) {
        std::cout << "CONVERGENCE CRITERION: Maximum "
          << "iteration number exceeded" << std::end;
      }

      return fx;
    }

    // Store optimization state from previous iterations
    fx_prev = fx; // Save the current objective function value
    xp.noalias() = x; // Save the current Beltrami coefficient (real format)
    mup.noalias() = mu; // Save the current Beltrami coefficient (complex format)
    wp.noalias() = w; // Save the current quasiconformal mapping
    map3Dp.noalias() = map3D; // Save the current 3D embedding
    gradp.noalias() = grad; // Save the current gradient vector (real format)

    // Perform line search to update unknowns
    try {

      // Once this procedure is finished, (fx, x, w, step) will be
      // updated, but grad will not
      MINGROCpp::LineSearchBacktracking<Scalar, Index>::LineSearch(
          *this, m_param, fixIDx, drt, dw, grad, fx, x, w, step);

    } catch ( const std::runtime_error &ere ) {

      // Re-set the current unknowns, gradients, and outputs
      // to their previous (valid) values
      w.noalias() = wp;
      mu.noalias() = mup;
      map3D.noalias() = map3Dp;

      throw;

    } catch ( const std::logic_error &ele ) {

      // Re-set the current unknowns, gradients, and outputs
      // to their previous (valid) values
      w.noalias() = wp;
      mu.noalias() = mup;
      map3D.noalias() = map3Dp;

      throw;

    }

    // Calculate the mapping kernel
    calculateMappingKernel( w, G1, G2, G3, G4 );

    // Update the Beltrami coefficient
    if (m_param.recomputeMu)
    {

      mu = m_F2V * ((m_Dc * w).array() / (m_Dz * w).array()).matrix();
      x << mu.real(), mu.imag();
      
    } else {

      convertRealToComplex(x, mu);

    }

    // Evaluate the function and gradient for initial configuration
    fx = calculateEnergyAndGrad(mu, w, G1, G2, G3, G4, gradMu, map3D, gamma);
    grad << gradMu.real(), gradMu.imag();
      
    // New vector norms
    xnorm = x.norm(); 
    gnorm = grad.norm();

    if ( m_param.iterDisp )
    {
      std::cout << "(" << std::setw(2) << iterNum << ")"
        << " f(x) = " << std::setw(15) << std::setprecision(10) << fx 
        << " ||x|| = " << std::setprecision(4) << std::setw(6) << xnorm
        << " ||dx|| = " << std::setw(10) << std::setprecision(7) << gnorm
        << " |df| = " << std::setw(10) << std::setprecision(7) << std::abs(fx-fx_prev);
    }

    // CONVERGENCE TEST -- Gradient
    if ( gnorm <= m_param.epsilon || gnorm <= m_param.epsilonRel * xnorm )
    {
      if ( m_param.iterDisp )
        std::cout << "CONVERGENCE CRITERION: Gradient norm" << std::endl;

      return fx;
    }

    // CONVERGENCE TEST -- Objective function value
    if ( fpast > 0 )
    {
      const Scalar fxd = fxp[iterNum % fpast];
      Scalar smallChange = m_param.delta * std::max(std::max(abs(fx), abs(fxd)), Scalar(1.0));

      bool longEnough = iterNum >= fpast;
      bool slowChange = std::abs(fxd-fx) <= smallChange;

      if( longEnough && slowChange )
      {
        if ( m_param.iterDisp )
          std::cout << "CONVERGENCE CRITERION: Insufficient change" << std::endl

        return fx;
      }

      fxp[iterNum % fpast] = fx;
    }

    // Update s and y
    // s_{k+1} = x_{k+1} - x_k
    // y_{k+1} = g_{k+1} - g_k
    bfgs.addCorrection( x - xp, grad - gradp );

    // Recursive formula to compute the new step direction d = -H * g
    bfgs.applyHv( grad, -Scalar(1.0), drt );

    // Convert real state format -> complex state format
    convertRealToComplex(drt, dmu);

    // Calculate the inital update direction for the quasiconformal mapping
    calculateMappingUpdate(dmu, G1, G2, G3, G4, dw);

    // Reset step = 1.0 as initial guess for the next line search
    step = Scalar(1.0);

  }
  
  return fx;

};

//TODO: Add explicit template instantiation
#ifdef MINGRO_STATIC_LIBRARY
#endif

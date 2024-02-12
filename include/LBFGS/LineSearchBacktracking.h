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

#ifndef _LINE_SEARCH_BACKTRACKING_H_
#define _LINE_SEARCH_BACKTRACKING_H_

#include "../MINGROC/mingrocInline.h"
#include "../external/NNIpp/include/NaturalNeighborInterpolant/NNIParam.h"
#include "../external/NNIpp/include/NaturalNeighborInterpolant/NaturalNeighborInterpolant.h"

#include <complex>
#include <Eigen/Core>

namespace MINGROCpp {

  ///
  /// The backtracking line search algorithm for L-BFGS
  /// Based on the implementation in 'LBFGS++' by Yixuan Qiu
  /// github.com/yixuan/LBFGSpp
  ///
  /// Templates:
  ///
  ///   Scalar    Input type of optimization unknowns
  ///   Index     Input type of 'MINGROC' indices
  ///
  template <typename Scalar, typename Index>
  class LineSearchBacktracking {

    private:

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

    public:

      ///
      /// Line search by backtracking
      ///
      /// Inputs:
      ///
      ///   mingro  An instance of the 'MINGROC' class. Used to evaluate the energy
      ///           and energy gradients
      ///
      ///   param   The parameters for the 'MINGROC' class
      ///
      ///   NNI     A natural neighbor interpolant for the final 3D surface
      ///
      ///   fixIDx  #FP by 1 list of fixed vertex IDs
      ///
      ///   drt     The current update direction
      ///
      ///   dw      The current update direction for the quasiconformal mapping
      ///
      ///   grad    The current global gradient vector
      ///
      ///   fx      The objective function value at the current point
      ///
      ///   x       The current global unknown vector
      ///
      ///   w       The current quasiconformal mapping
      ///
      ///   step    The initial step length
      ///
      ///   calcGrowthEnergy    Whether or not to calculate the areal growth energy.
      ///                       This is the only term in the energy that depends on
      ///                       the BHF
      ///
      ///   calcMuEnergy        Whether or not to calculate the terms in the energy
      ///                       that depend on the Beltrami coefficient directly,
      ///                       i.e. not through the mapping w
      ///
      /// Outputs:
      ///
      ///   fx          The objective function value at the updated point
      ///
      ///   x           The updated global unknown vector
      ///
      ///   w           The updated quasiconformal mapping
      ///
      ///   step        The calculated step length
      ///
      static void LineSearch(
          const MINGROC<Scalar, Index> &mingroc,
          const MINGROCParam<Scalar> &param,
          const NNIpp::NaturalNeighborInterpolant<Scalar> &NNI,
          const Eigen::Matrix<Index, Eigen::Dynamic, 1> &fixIDx,
          const Vector &drt, const CplxVector &dw, const Vector &grad,
          bool calcGrowthEnergy, bool calcMuEnergy,
          Scalar &fx, Vector &x, CplxVector &w, Scalar &step );

  };

}

#ifndef MINGROC_STATIC_LIBRARY
#  include "LineSearchBacktracking.cpp"
#endif

#endif

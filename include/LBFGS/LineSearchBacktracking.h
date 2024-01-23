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

#include "../MINGROCpp/mingroInline.h"
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
      typedef Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic> Array;
      typedef Eigen::Matrix<CScalar, Eigen::Dynamic, 1> CplxVector;

    public:

      ///
      /// Line search by backtracking
      ///
      /// Inputs:
      ///
      ///   mingro  An instance of the 'MINGROC' class. Used to evaluate the energy
      ///           and energy gradients
      ///
      ///   param   The parameters for the 'DSEM' class
      ///
      ///   L       #E by 1 list of target edge lengths
      ///
      ///   tarA_E  #E by 1 list of target areas associated to edges
      ///
      ///   tarA_V  #V by 1 list of target areas associated to vertices
      ///
      ///   fx      The objective function value at the current point
      ///
      ///   grad    The current global gradient vector
      ///
      ///   step    The initial step length
      ///
      ///   xp      The current global unknown vector
      ///
      ///   wp      The current quasiconformal mapping
      ///
      ///   drt     The current update direction
      ///
      ///   dw      The current update direction for the quasiconformal mapping
      ///
      /// Outputs:
      ///
      ///   fx          The objective function value at the updated point
      ///
      ///   x           The updated global unknown vector
      ///
      ///   w           The updated quasiconformal mapping
      ///
      ///   grad        The updated global gradient vector
      ///
      ///   gradUNorm   The norm of the gradient with respect to the fully
      ///               composed parameterization
      ///
      ///   step        The calculated step length
      ///
      ///   l           The updated metric associated with the minimum distance SEM
      ///
      static void LineSearch(
          const MINGROC<Scalar, Index> &mingro, const MINGROCParam<Scalar> &param,
          const Vector &L, const Vector &tarA_E, const Vector &tarA_V,
          const Vector &xp, const CplxVector &wp,
          const Vector &drt, const CplxVector &dw,
          Scalar &fx, Vector &x, CplxVector &w,
          Vector &grad, Scalar &gradUNorm, Scalar &step, Vector &l );

  };

}

#ifndef MINGROC_STATIC_LIBRARY
#  include "LineSearchBacktracking.cpp"
#endif

#endif

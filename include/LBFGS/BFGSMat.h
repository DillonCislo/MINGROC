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

#ifndef _BFGS_MAT_H_
#define _BFGS_MAT_H_

#include "../MINGROC/mingrocInline.h"
#include <Eigen/Core>
#include <vector>

namespace MINGROCpp {

  ///
  /// An implicit representation of the BFGS approximation to the Hessian matrix B
  ///
  /// B = theta * I - W * M * W'
  /// H = inv(B)
  ///
  /// Based on the implementation in 'LBFGS++' by Yixuan Qiu
  /// github.com/yixuan/LBFGSpp
  ///
  /// Reference:
  ///
  /// [1] D. C. Liu and J. Nocedal (1989). On the limited memory BFGS method for
  /// large scale optimization
  ///
  /// Templates:
  ///
  ///   Scalar    Input type of optimization unknowns
  ///
  template <typename Scalar>
  class BFGSMat {

    private:

      typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
      typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
      typedef Eigen::Ref<const Vector> RefConstVec;
      typedef std::vector<int> IndexSet;

      // The maximum number of correction vectors
      int m_m;

      // theta * I is the initial approximation to the Hessian matrix
      Scalar m_theta;

      // History of the s vectors
      Matrix m_s;

      // History of the y vectors
      Matrix m_y;

      // History of the s'y values
      Vector m_ys;

      // Temporary values used in computing H * v
      Vector m_alpha;

      // Number of correction vectors in the history, m_ncorr <= m
      int m_ncorr;

      // A pointer to locate the most recent history, 1 <= m_ptr <= m
      // Details: s and y are vectors stored in cyclic order.
      //    For example, if the current s-vector is sotred in m_s[, m-1],
      //    then in the next iteration m_s[, 0] will be overwritten.
      //    m_s[, m_ptr-1] points to the most recent history and
      //    m_s[, m_ptr % m] points to the most distant one
      int m_ptr;

    public:

      ///
      /// Null constructor
      ///
      BFGSMat() {};

      ///
      /// Reset internal variables
      ///
      /// Inputs:
      ///
      ///   n     Dimension of the vector to be optimized
      ///   m     Maximum number of corrections to approximate the Hessian matrix
      ///
      MINGROC_INLINE void reset( int n, int m );

      ///
      /// Add correction vectors to the BFGS matrix
      ///
      /// Inputs:
      ///
      ///   s     New s vector from current optimization iteration
      ///   y     New y vector from current optimization iteration
      ///
      MINGROC_INLINE void addCorrection( const RefConstVec &s, const RefConstVec &y );

      ///
      /// Recursive formula to compute a * H * v, where a is a scalar and v is an
      /// (n x 1) vector. H0 = (1/theta) * I is the initial approximation to H.
      /// Algorithm 7.4 of Nocedal, J. & Wright, S. (2006), Numerical Optimization
      ///
      /// Inputs:
      ///
      ///   v     #V by 1 vector
      ///   a     A scalar value
      ///
      /// Outputs:
      ///
      ///   res   #V by 1 vector
      ///
      MINGROC_INLINE void applyHv( const Vector &v, const Scalar &a, Vector &res );

  };

}

#ifndef MINGROC_STATIC_LIBRARY
#  include "BFGSMat.cpp"
#endif

#endif

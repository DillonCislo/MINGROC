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

#ifndef _CLIP_TO_UNIT_CIRCLE_H_
#define _CLIP_TO_UNIT_CIRCLE_H_

#include "mingrocInline.h"

#include <complex>
#include <Eigen/Core>

namespace MINGROCpp {

  ///
  /// Clip a set of 2D points to all lie on the unit circle
  ///
  /// Templates:
  ///
  ///   Scalar    Input type of data points
  ///
  /// Inputs:
  ///
  ///   X   #P by 2 list of 2D point coordinates
  ///
  /// Outputs:
  ///
  ///   X   #P by 2 list of clipped 2D point coordinates
  ///
  template <typename Scalar>
  MINGROC_INLINE void clipToUnitCircle(
      Eigen::Matrix<Scalar, Eigen::Dynamic, 2> &X );

  ///
  /// Clip a set of complex numbers to all lie on the unit circle
  ///
  /// Templates:
  ///
  ///   Scalar    Input type of data points
  ///
  /// Inputs:
  ///
  ///   X   #P by 1 list of complex numbers
  ///
  /// Outputs:
  ///
  ///   X   #P by 1 list of clipped complex numbers
  ///
  template <typename Scalar>
  MINGROC_INLINE void clipToUnitCircle(
      Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, 1> &X );

  ///
  /// Clip a subset of a set of 2D points to all lie on the unit circle
  ///
  /// Templates:
  ///
  ///   Scalar    Input type of data points
  ///   Index     Input type of point indices
  ///
  /// Inputs:
  ///
  ///   vID       #CP by 1 list of point IDs to clip
  ///   X         #P by 2 list of 2D point coordinates
  ///
  /// Outputs:
  ///
  ///   X         #P by 2 list of clipped 2D point coordinates
  ///
  template <typename Scalar, typename Index>
  MINGROC_INLINE void clipToUnitCircle(
      const Eigen::Matrix<Index, Eigen::Dynamic, 1> &vID,
      Eigen::Matrix<Scalar, Eigen::Dynamic, 2> &X );

  ///
  /// Clip a subset of a set of complex numbers to all lie on the unit circle
  ///
  /// Templates:
  ///
  ///   Scalar    Input type of data points
  ///   Index     Input type of point indices
  ///
  /// Inputs:
  ///
  ///   vID       #CP bt 1 list of point IDs to clip
  ///   X         #P by 1 list of complex numbers
  ///
  /// Outputs:
  ///
  ///   X         #P by 1 list of clipped complex numbers
  ///
  template <typename Scalar, typename Index>
  MINGROC_INLINE void clipToUnitCircle(
      const Eigen::Matrix<Index, Eigen::Dynamic, 1> &vID,
      Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, 1> &X );

};

#ifndef MINGROC_STATIC_LIBRARY
#  include "clipToUnitCircle.cpp"
#endif

#endif

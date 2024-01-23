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


#include <cmath>

///
/// Clip a set of 2D points to all lie on the unit circle
///
template <typename Scalar>
MINGROC_INLINE void DSEMpp::clipToUnitCircle(
    Eigen::Matrix<Scalar, Eigen::Dynamic, 2> &X ) {

  typedef std::complex<Scalar> CScalar;

  for( int k = 0; k < X.rows(); k++ ) {

    CScalar curX(X(k,0), X(k,1));
    curX = std::exp( CScalar(Scalar(0.0), std::arg(curX)) );

    X(k,0) = curX.real();
    X(k,1) = curX.imag();

  }

};

///
/// Clip a set of complex numbers to all lie on the unit circle
///
template <typename Scalar>
MINGROC_INLINE void DSEMpp::clipToUnitCircle(
    Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, 1> &X ) {

  typedef std::complex<Scalar> CScalar;

  for( int k = 0; k < X.size(); k++ ) {

    X(k) = std::exp( CScalar(Scalar(0.0), std::arg(X(k))) );

  }

};

///
/// Clip a subset of a set of 2D points to all lie on the unit circle
///
template <typename Scalar, typename Index>
MINGROC_INLINE void DSEMpp::clipToUnitCircle(
    const Eigen::Matrix<Index, Eigen::Dynamic, 1> &vID,
    Eigen::Matrix<Scalar, Eigen::Dynamic, 2> &X ) {

  typedef std::complex<Scalar> CScalar;

  for( int k = 0; k < vID.size(); k++ ) {

    CScalar curX(X(vID(k), 0), X(vID(k), 1));
    curX = std::exp( CScalar(Scalar(0.0), std::arg(curX)) );

    X(vID(k), 0) = curX.real();
    X(vID(k), 1) = curX.imag();

  }

};

///
/// Clip a subset of a set of complex numbers to all lie on the unit circle
///
template <typename Scalar, typename Index>
MINGROC_INLINE void DSEMpp::clipToUnitCircle(
    const Eigen::Matrix<Index, Eigen::Dynamic, 1> &vID,
    Eigen::Matrix<std::complex<Scalar>, Eigen::Dynamic, 1> &X ) {

  typedef std::complex<Scalar> CScalar;

  for( int k = 0; k < vID.size(); k++ ) {

    X(vID(k)) = std::exp( CScalar(Scalar(0.0), std::arg(X(vID(k)))) );

  }

};

// TODO: Add explicit template instantiation
#ifdef MINGROC_STATIC_LIBRARY
#endif

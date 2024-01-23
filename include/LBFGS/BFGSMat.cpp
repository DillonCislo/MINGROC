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

#include "BFGSMat.h"

///
/// Reset internal variables
///
template <typename Scalar>
MINGROC_INLINE void MINGROCpp::BFGSMat<Scalar>::reset( int n, int m ) {

  m_m = m;
  m_theta = Scalar(1.0);
  m_s.resize(n, m);
  m_y.resize(n, m);
  m_ys.resize(m);
  m_alpha.resize(m);
  m_ncorr = 0;
  m_ptr = m; // This makes sure that m_ptr % m == 0 in the first step 

};

///
/// Add correction vectors to the BFGS matrix
///
template <typename Scalar>
MINGROC_INLINE void MINGROCpp::BFGSMat<Scalar>::addCorrection(
    const RefConstVec &s, const RefConstVec &y ) {

  const int loc = m_ptr % m_m;

  m_s.col(loc).noalias() = s;
  m_y.col(loc).noalias() = y;

  // ys = y's = 1/rho
  const Scalar ys = m_s.col(loc).dot(m_y.col(loc));
  m_ys[loc] = ys;

  m_theta = m_y.col(loc).squaredNorm() / ys;

  if (m_ncorr < m_m) { m_ncorr++; }

  m_ptr = loc + 1;

};

///
/// Recursive formula to compute a * H * v, where a is a scalar and v is an
/// (n x 1) vector. H0 = (1/theta) * I is the initial approximation to H.
/// Algorithm 7.4 of Nocedal, J. & Wright, S. (2006), Numerical Optimization
///
template <typename Scalar>
MINGROC_INLINE void MINGROCpp::BFGSMat<Scalar>::applyHv(
    const Vector &v, const Scalar &a, Vector &res ) {

  res.resize(v.size());

  // L-BFGS two-loop recursion
  
  // Loop 1
  res.noalias() = a * v;
  int j = m_ptr % m_m;
  for( int i = 0; i < m_ncorr; i++ ) {

    j = (j + m_m - 1) % m_m;
    m_alpha[j] = m_s.col(j).dot(res) / m_ys[j];
    res.noalias() -= m_alpha[j] * m_y.col(j);

  }

  // Apply initial H0
  res /= m_theta;

  // Loop 2
  for( int i = 0; i < m_ncorr; i++ ) {

    const Scalar beta = m_y.col(j).dot(res) / m_ys[j];
    res.noalias() += (m_alpha[j] - beta) * m_s.col(j);
    j = (j + 1) % m_m;

  }

};

// TODO: Add explicit template instantiation
#ifdef MINGROC_STATIC_LIBRARY
#endif

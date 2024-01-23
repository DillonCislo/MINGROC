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

#include "MINGROCParam.h"

#include <cassert>
#include <stdexcept>
#include <iostream>

///
/// Constructor for parameter class
///
template <typename Scalar>
MINGROCpp::MINGROCParam<Scalar>::MINGROCParam() {

  m = 6;
  epsilon = Scalar(1e-5);
  epsilonRel = Scalar(1e-5);
  past = 0;
  delta = 0;
  maxIterations = 0;
  lineSearchTermination = LINE_SEARCH_TERMINATION_ARMIJO;
  lineSearchMethod = LINE_SEARCH_BACKTRACKING;
  maxLineSearch = 20;
  minStep = Scalar(1e-20);
  maxStep = Scalar(1e20);
  ftol = Scalar(1e-4);
  wolfe = Scalar(0.9);
  CC = Scalar(0.0);
  SC = Scalar(0.0);
  DC = Scalar(0.0);
  iterDisp = false;
  iterVisDisp = false;
  checkSelfIntersections = true;
  recomputeMu = true;

};

///
/// Check the validity of the parameters
///
template <typename Scalar>
MINGROC_INLINE void MINGROCpp::MINGROCParam<Scalar>::checkParam() const {

  if ( m <= 0 )
    throw std::invalid_argument("'m' must be positive");
  if ( epsilon < 0 )
    throw std::invalid_argument("'epsilon' must be non-negative");
  if ( epsilonRel < 0 )
    throw std::invalid_argument("'epsilonRel' must be non-negative");
  if ( past < 0 )
    throw std::invalid_argument("'past' must be non-negative");
  if ( delta < 0 )
    throw std::invalid_argument("'delta' must be non-negative");
  if ( maxIterations < 0 )
    throw std::invalid_argument("'maxIterations' must be non-negative");
  if ( lineSearchTermination < LINE_SEARCH_TERMINATION_NONE ||
     lineSearchTermination > LINE_SEARCH_TERMINATION_ARMIJO )
    throw std::invalid_argument("Unsupported line search termination condition");
  if ( lineSearchMethod < LINE_SEARCH_BACKTRACKING ||
      lineSearchMethod > LINE_SEARCH_MORE_THUENTE )
    throw std::invalid_argument("Unsupported line search method");
  if ( maxLineSearch <= 0 )
    throw std::invalid_argument("'maxLineSearch' must be positive");
  if ( minStep < 0 ) 
    throw std::invalid_argument("'minStep' must be positive");
  if ( maxStep < minStep )
    throw std::invalid_argument("'maxStep' must be greater than 'minStep'");
  if ( ftol <= 0 || ftol >= 0.5 )
    throw std::invalid_argument("'ftol' must satisfy 0 < ftol < 0.5");
  if ( wolfe <= ftol || wolfe >= 1 )
    throw std::invalid_argument("'wolfe' must satisfy ftol < wolfe < 1");
  if ( CC < 0 )
    throw std::invalid_argument("'CC' must be non-negative");
  if ( SC < 0 )
    throw std::invalid_argument("'SC' must be non-negative");
  if ( DC < 0 )
    throw std::invalid_argument("'DC' must be non-negative");

};

// TODO: Add explicit template instantiations
#ifdef MINGROC_STATIC_LIBRARY
#endif

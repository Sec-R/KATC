#pragma once
#include "semilinear.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"

// Convert a SemilinearSet (the Parikh image of a regex over an alphabet of
// size k) to an MLIR PresburgerSet.
//
// Each linear set  L = b + <p_1,...,p_m>  becomes one IntegerPolyhedron
// with:
//   - k  dim variables  x_0,...,x_{k-1}  (the Parikh vector components)
//   - m  local variables n_0,...,n_{m-1} (existential, one per period)
//
// Constraints:
//   equalities  : x_i = b_i + sum_j n_j * p_j[i]    for each i in [0,k)
//   inequalities: n_j >= 0                            for each j in [0,m)
//
// The resulting PresburgerSet represents:
//   { (x_0,...,x_{k-1}) in Z^k | exists n_0,...,n_{m-1} in N.
//       x_i = b_i + sum_j n_j*p_j[i]  for all i }
//
// The union over all linear sets gives the full Parikh image.
mlir::presburger::PresburgerSet to_presburger_set(const SemilinearSet& S, int k);

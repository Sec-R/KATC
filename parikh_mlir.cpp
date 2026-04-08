#include "parikh_mlir.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir::presburger;
using llvm::SmallVector;

// Build one IntegerPolyhedron for a single linear set  b + <p_1,...,p_m>.
//
// Variable layout in every constraint vector:
//   [ x_0, ..., x_{k-1},  n_0, ..., n_{m-1},  constant ]
//    |<--- k dim vars --->|<--- m local vars -->|
//
// Constraint semantics (MLIR convention):
//   equality   : dot(coeff, [vars; 1]) = 0
//   inequality : dot(coeff, [vars; 1]) >= 0
static IntegerPolyhedron linear_set_to_poly(const LinearSet& L, int k) {
    int m = (int)L.periods.size();

    // k dim variables, 0 symbols, m local (existential) variables.
    PresburgerSpace space = PresburgerSpace::getSetSpace(k, /*numSymbols=*/0, /*numLocals=*/m);
    IntegerPolyhedron poly(space);

    int numVars = k + m;  // dims + locals (MLIR appends constant implicitly)

    // Equality for each dimension i:
    //   x_i - b_i - sum_j n_j * p_j[i] = 0
    for (int i = 0; i < k; i++) {
        SmallVector<int64_t> eq(numVars + 1, 0);
        eq[i]       = 1;              // coefficient of x_i
        for (int j = 0; j < m; j++)
            eq[k + j] = -L.periods[j][i]; // coefficient of n_j (negated)
        eq[numVars] = -L.base[i];    // constant term (negated b_i)
        poly.addEquality(eq);
    }

    // Non-negativity inequality for each local variable n_j >= 0:
    //   n_j >= 0  <=>  0*x_0 + ... + 1*n_j + ... + 0 >= 0
    for (int j = 0; j < m; j++) {
        SmallVector<int64_t> ineq(numVars + 1, 0);
        ineq[k + j] = 1;
        poly.addInequality(ineq);
    }

    return poly;
}

PresburgerSet to_presburger_set(const SemilinearSet& S, int k) {
    // The set space has only the k dim variables (locals are per-disjunct).
    PresburgerSpace setSpace = PresburgerSpace::getSetSpace(k);
    PresburgerSet result = PresburgerSet::getEmpty(setSpace);

    for (const LinearSet& L : S)
        result.unionInPlace(linear_set_to_poly(L, k));

    return result;
}

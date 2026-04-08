#pragma once
#include "regex.h"
#include "nfa.h"
#include <cstdint>
#include <vector>

using Vec = std::vector<int64_t>;

// A linear set  b + <p_1,...,p_m>  =  { b + sum n_i p_i | n_i in N }
struct LinearSet {
    Vec base;
    std::vector<Vec> periods;
};

// A semilinear set is a finite union of linear sets.
using SemilinearSet = std::vector<LinearSet>;

// Compute the Parikh image P(e) structurally:
//   P(0)   = empty
//   P(eps) = { 0^k }
//   P(a_i) = { e_i }
//   P(r+s) = P(r) union P(s)
//   P(rs)  = P(r) Minkowski-sum P(s)
//   P(r*)  = star(P(r))  = 0 + < all bases and periods of P(r) >
//
// num_letters = |Sigma| = dimension k of the Parikh vectors.
SemilinearSet parikh_image(const RegexPtr& e, int num_letters);

// Check whether vector v is in S (for testing; uses bounded backtracking).
bool sl_contains(const SemilinearSet& S, const Vec& v);

// Compute the Parikh image of the language accepted by an NFA.
// Uses Kleene's path algorithm on semilinear matrices:
//   R[i][j] = semilinear set of Parikh vectors of all paths from state i to j.
// Returns the union over all (initial, final) pairs.
SemilinearSet parikh_image_nfa(const NFA& nfa, int num_letters);

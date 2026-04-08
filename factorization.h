#pragma once
#include "nfa.h"
#include "regex.h"
#include <set>
#include <vector>
#include <utility>

// Partition of alphabet {0,...,num_letters-1} into m equivalence classes.
// Letters in the same class mutually commute.
struct Partition {
    int num_letters;                           // |Σ|
    int num_classes;                           // m
    std::vector<int> letter_to_class;          // letter  → class index (0-based)
    std::vector<std::vector<int>> class_letters; // class  → list of letters
};

// A single transition of the factorization automaton.
// Label is a RegexPtr representing A_{k2}^+[q1][q2]:
// the non-empty words over Σ_{k2} that take q1→q2.
struct FacTransition {
    int      from;   // encode(q1, k1)
    RegexPtr label;
    int      to;     // encode(q2, k2)
};

struct FactorizationAutomaton {
    int num_orig_states;
    int num_classes;
    std::set<int> initial;
    std::set<int> final_states;
    std::vector<FacTransition> transitions;

    // Encode (original state q, class index k) as a single integer
    int encode(int q, int k) const { return q * num_classes + k; }
    // Decode back
    std::pair<int,int> decode(int s) const { return {s / num_classes, s % num_classes}; }
    int total_states() const { return num_orig_states * num_classes; }
};

// Build M_fac from a standard NFA and a commutativity partition.
//
// Algorithm:
//   For each destination class k2:
//     1. Build A_{k2}: n×n RegexPtr matrix where A_{k2}[i][j] =
//        re_union of re_letter(a) for each a ∈ Σ_{k2} with transition (i,a,j).
//     2. Compute A_{k2}^+ = A_{k2} · A_{k2}^*  (one-or-more, via matrix_star).
//     3. For each (q1,q2) with A_{k2}^+[q1][q2] ≠ re_zero(),
//        and for every source class k1 ≠ k2,
//        add transition ((q1,k1), A_{k2}^+[q1][q2], (q2,k2)).
FactorizationAutomaton build_factorization_automaton(const NFA& nfa,
                                                      const Partition& part);

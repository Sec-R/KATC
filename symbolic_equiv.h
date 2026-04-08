#pragma once
#include "parikh_mlir.h"
#include "factorization.h"
#include <set>
#include <vector>

// A single transition of a symbolic automaton.
// The label is a PresburgerSet over Z^dim: the transition fires for any
// input vector v that belongs to the label.
struct SymTransition {
    int from;
    mlir::presburger::PresburgerSet label;
    int to;
};

// Symbolic automaton over Z^dim.
// A word is a finite sequence of vectors in Z^dim.
// Acceptance: exists a run where each step's vector lies in the transition label.
struct SymbolicAutomaton {
    int num_states = 0;
    int dim = 0;  // dimension of the label domain (= |Σ|)
    std::set<int> initial;
    std::set<int> final_states;
    std::vector<SymTransition> transitions;
};

// Convert a FactorizationAutomaton to a SymbolicAutomaton by replacing each
// NFA transition label with its Parikh image (as a PresburgerSet over Z^num_letters).
SymbolicAutomaton fac_to_symbolic(const FactorizationAutomaton& fac,
                                   int num_letters);

// Compute the minterms (satisfiable Boolean atoms) of the given PresburgerSets.
//
// Uses a partition-refinement approach:
//   Start with {universe}.  For each φ_i, split every current atom A into
//   (A ∩ φ_i) and (A \ φ_i), discarding empty pieces.
//
// Result: a set of non-empty, pairwise-disjoint PresburgerSets whose union
// is the domain Z^dim.  Every original φ_i is a (possibly non-disjoint)
// union of some minterms.
std::vector<mlir::presburger::PresburgerSet>
compute_minterms(const std::vector<mlir::presburger::PresburgerSet>& sets,
                 int dim);

// Check whether two symbolic automata over the same dimension accept the
// same language (i.e., the same set of finite sequences of Z^dim vectors).
//
// Algorithm:
//   1. Collect all transition labels from both automata.
//   2. Compute minterms: each minterm acts as an atomic "letter".
//   3. Determinize both automata via powerset construction over the minterms:
//        post(S, μ) = { q' | ∃q∈S, ∃ q–φ–>q' with μ ⊆ φ }
//   4. BFS from (initial_1, initial_2) over pairs of powerset states,
//      checking that acceptance conditions agree at every reachable pair
//      (Hopcroft-Karp style bisimulation).
bool check_equiv(const SymbolicAutomaton& a1, const SymbolicAutomaton& a2);

// ── Word acceptance ───────────────────────────────────────────────────────────

// Parse a word into a sequence of C-blocks (one Parikh vector per block).
//
// Consecutive letters of the same equivalence class are grouped into one block.
// Each block is converted to its Parikh vector in Z^{num_letters}.
//
// Example: word=[a,b,c,b,a], class_0={a,b}, class_1={c}
//   → blocks [ (1,1,0), (0,0,1), (1,1,0) ]
std::vector<Vec> word_to_blocks(const std::vector<int>& word,
                                const Partition& part);

// Test whether a symbolic automaton accepts a sequence of Parikh vectors.
//
// Simulates the automaton non-deterministically (powerset):
//   for each block vector v, advance every current state q via any transition
//   (q, φ, q') where v ∈ φ  (checked with containsPoint).
// Returns true iff the final powerset intersects the final states.
bool sym_accepts(const SymbolicAutomaton& sa, const std::vector<Vec>& blocks);

// Full-pipeline convenience function: build the symbolic automaton for regex e
// under partition part, then test whether word belongs to the C-language.
bool katc_accepts(const RegexPtr& e, const Partition& part,
                  const std::vector<int>& word);

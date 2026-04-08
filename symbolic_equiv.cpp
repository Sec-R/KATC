#include "symbolic_equiv.h"
#include "regex.h"
#include "semilinear.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "llvm/ADT/ArrayRef.h"
#include <cassert>
#include <map>
#include <queue>

using namespace mlir::presburger;

// ── fac_to_symbolic ───────────────────────────────────────────────────────────

SymbolicAutomaton fac_to_symbolic(const FactorizationAutomaton& fac,
                                   int num_letters) {
    SymbolicAutomaton sa;
    sa.num_states   = fac.total_states();
    sa.dim          = num_letters;
    sa.initial      = fac.initial;
    sa.final_states = fac.final_states;

    for (const FacTransition& t : fac.transitions) {
        SemilinearSet sl = parikh_image(t.label, num_letters);
        if (sl.empty()) continue;
        PresburgerSet ps = to_presburger_set(sl, num_letters);
        if (ps.isIntegerEmpty()) continue;
        sa.transitions.push_back({t.from, std::move(ps), t.to});
    }
    return sa;
}

// ── compute_minterms ──────────────────────────────────────────────────────────
//
// Partition-refinement starting from the LABELS themselves (not Z^dim).
//
// Points that lie outside every label map all powerset states to {} in both
// automata and are trivially bisimilar — we never need to visit them.
// Starting from the labels avoids computing Presburger complements of the
// infinite universe, which is very expensive.
//
// Invariant after processing each label: the current atoms are pairwise
// disjoint and every atom is a subset of at least one original label.
// Each original label is a (not necessarily disjoint) union of some atoms.

std::vector<PresburgerSet>
compute_minterms(const std::vector<PresburgerSet>& labels, int dim) {
    std::vector<PresburgerSet> atoms;
    atoms.reserve(labels.size());

    for (const PresburgerSet& phi : labels) {
        if (phi.isIntegerEmpty()) continue;

        // phi_rest = part of phi not yet represented by any existing atom.
        PresburgerSet phi_rest = phi;

        std::vector<PresburgerSet> next;
        for (size_t idx = 0; idx < atoms.size(); ++idx) {
            const PresburgerSet& a = atoms[idx];
            PresburgerSet inter = a.intersect(phi_rest);
            if (inter.isIntegerEmpty()) {
                // a and phi_rest are disjoint: a is unchanged.
                next.push_back(a);
                continue;
            }
            // a overlaps phi_rest.  Split a into (a ∩ phi_rest) and (a \ phi_rest).
            next.push_back(inter);                      // inside phi
            PresburgerSet a_out = a.subtract(phi_rest);
            if (!a_out.isIntegerEmpty())
                next.push_back(std::move(a_out));       // outside phi
            // Remove the covered portion from phi_rest.
            phi_rest = phi_rest.subtract(a);
            if (phi_rest.isIntegerEmpty()) {
                // Remaining atoms are unaffected — copy them as-is.
                for (size_t j = idx + 1; j < atoms.size(); ++j)
                    next.push_back(atoms[j]);
                break;
            }
        }
        if (!phi_rest.isIntegerEmpty())
            next.push_back(std::move(phi_rest));        // part of phi with no overlap

        atoms = std::move(next);
    }
    return atoms;
}

// ── Powerset-construction helpers ─────────────────────────────────────────────

// For a true minterm μ (produced by compute_minterms), μ is either entirely
// inside φ or entirely outside φ.  Therefore:
//   μ ⊆ φ  ↔  μ ∩ φ ≠ ∅
// Using intersect instead of subtract avoids expensive complement computation.
static bool minterm_subset_of(const PresburgerSet& mu,
                               const PresburgerSet& phi) {
    return !mu.intersect(phi).isIntegerEmpty();
}

// post(S, μ): states reachable from S on any transition whose label contains μ.
static std::set<int> powerset_post(const std::set<int>&           S,
                                    const PresburgerSet&            mu,
                                    const std::vector<SymTransition>& trans) {
    std::set<int> result;
    for (const SymTransition& t : trans)
        if (S.count(t.from) && minterm_subset_of(mu, t.label))
            result.insert(t.to);
    return result;
}

// A powerset state is final iff it contains at least one original final state.
static bool powerset_is_final(const std::set<int>& S,
                               const std::set<int>& finals) {
    for (int s : S)
        if (finals.count(s)) return true;
    return false;
}

// ── check_equiv ───────────────────────────────────────────────────────────────

bool check_equiv(const SymbolicAutomaton& a1, const SymbolicAutomaton& a2) {
    assert(a1.dim == a2.dim);
    int dim = a1.dim;

    // Collect all transition labels from both automata.
    std::vector<PresburgerSet> all_labels;
    for (const SymTransition& t : a1.transitions) all_labels.push_back(t.label);
    for (const SymTransition& t : a2.transitions) all_labels.push_back(t.label);

    // Compute minterms.
    std::vector<PresburgerSet> minterms = compute_minterms(all_labels, dim);

    // BFS over pairs (S1, S2) of powerset states.
    // S1 ⊆ Q1, S2 ⊆ Q2.
    using PowerState = std::set<int>;
    using PowerPair  = std::pair<PowerState, PowerState>;

    std::map<PowerPair, bool> visited;
    std::queue<PowerPair>     worklist;

    worklist.push({a1.initial, a2.initial});

    while (!worklist.empty()) {
        auto [S1, S2] = worklist.front();
        worklist.pop();

        if (visited.count({S1, S2})) continue;
        visited[{S1, S2}] = true;

        // Acceptance condition must agree.
        bool f1 = powerset_is_final(S1, a1.final_states);
        bool f2 = powerset_is_final(S2, a2.final_states);
        if (f1 != f2) return false;

        // For each minterm, successors must also be bisimilar.
        for (const PresburgerSet& mu : minterms) {
            PowerState T1 = powerset_post(S1, mu, a1.transitions);
            PowerState T2 = powerset_post(S2, mu, a2.transitions);
            if (!visited.count({T1, T2}))
                worklist.push({T1, T2});
        }
    }
    return true;
}

// ── Word acceptance ───────────────────────────────────────────────────────────

std::vector<Vec> word_to_blocks(const std::vector<int>& word,
                                const Partition& part) {
    std::vector<Vec> blocks;
    int n = (int)word.size();
    int i = 0;
    while (i < n) {
        int cls = part.letter_to_class[word[i]];
        Vec v(part.num_letters, 0);
        while (i < n && part.letter_to_class[word[i]] == cls) {
            v[word[i]] += 1;
            ++i;
        }
        blocks.push_back(std::move(v));
    }
    return blocks;
}

bool sym_accepts(const SymbolicAutomaton& sa, const std::vector<Vec>& blocks) {
    std::set<int> current = sa.initial;

    for (const Vec& v : blocks) {
        // llvm::ArrayRef requires a contiguous int64_t array; Vec = vector<int64_t>.
        llvm::ArrayRef<int64_t> point(v.data(), v.size());

        std::set<int> next;
        for (const SymTransition& t : sa.transitions) {
            if (current.count(t.from) && t.label.containsPoint(point))
                next.insert(t.to);
        }
        current = std::move(next);
    }

    // Accept iff current powerset intersects final states.
    for (int s : current)
        if (sa.final_states.count(s)) return true;
    return false;
}

bool katc_accepts(const RegexPtr& e, const Partition& part,
                  const std::vector<int>& word) {
    NFA nfa = regex_to_nfa(e, part.num_letters);
    FactorizationAutomaton fac = build_factorization_automaton(nfa, part);
    SymbolicAutomaton sa = fac_to_symbolic(fac, part.num_letters);
    std::vector<Vec> blocks = word_to_blocks(word, part);
    return sym_accepts(sa, blocks);
}

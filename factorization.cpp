#include "factorization.h"

FactorizationAutomaton build_factorization_automaton(const NFA& nfa,
                                                      const Partition& part) {
    const int n = nfa.num_states;
    const int m = part.num_classes;

    FactorizationAutomaton fac;
    fac.num_orig_states = n;
    fac.num_classes     = m;

    // Initial states: Q_0 × {0,...,m-1}
    for (int q : nfa.initial)
        for (int k = 0; k < m; k++)
            fac.initial.insert(fac.encode(q, k));

    // Final states: F × {0,...,m-1}
    for (int q : nfa.final_states)
        for (int k = 0; k < m; k++)
            fac.final_states.insert(fac.encode(q, k));

    // Semiring over RegexPtr
    MatSR<RegexPtr> re_sr {
        re_zero(),
        [](RegexPtr a, RegexPtr b) { return re_union(a, b); },
        [](RegexPtr a, RegexPtr b) { return re_concat(a, b); },
        [](RegexPtr a)             { return re_star(a); }
    };

    // For each destination class k2, compute A_{k2}^+ and emit transitions.
    for (int k2 = 0; k2 < m; k2++) {

        // Step 1: build the regex matrix restricted to Σ_{k2}
        // A_k2[i][j] = re_union of re_letter(a) for each a ∈ Σ_{k2} with (i,a,j) ∈ Δ
        NfaMatrix<RegexPtr> A_k = mat_zero<RegexPtr>(n, n, re_zero());
        for (int i = 0; i < n; i++) {
            for (auto& [letter, nexts] : nfa.trans[i]) {
                if (letter == EPSILON) continue;
                if (part.letter_to_class[letter] != k2) continue;
                for (int j : nexts)
                    A_k[i][j] = re_union(A_k[i][j], re_letter(letter));
            }
        }

        // Step 2: A_{k2}^+ = A_{k2} · A_{k2}^*
        NfaMatrix<RegexPtr> A_k_star = matrix_star(A_k, re_sr);
        NfaMatrix<RegexPtr> A_k_plus = mat_mul(A_k, A_k_star, re_sr);

        // Step 3: emit transitions
        // ((q1, k1), A_{k2}^+[q1][q2], (q2, k2))  for all k1 ≠ k2
        for (int q1 = 0; q1 < n; q1++) {
            for (int q2 = 0; q2 < n; q2++) {
                if (A_k_plus[q1][q2]->kind == Regex::Kind::Zero) continue;
                for (int k1 = 0; k1 < m; k1++) {
                    if (k1 == k2) continue;
                    FacTransition t;
                    t.from  = fac.encode(q1, k1);
                    t.label = A_k_plus[q1][q2];
                    t.to    = fac.encode(q2, k2);
                    fac.transitions.push_back(t);
                }
            }
        }
    }

    return fac;
}

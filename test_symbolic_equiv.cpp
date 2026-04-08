// test_symbolic_equiv.cpp
//
// Tests for compute_minterms and check_equiv.
//
// Alphabet: 2 letters {a=0, b=1}.
// "Letter a" as a symbolic token = Parikh vector (1,0)
//           → PresburgerSet  {v | v[0]=1, v[1]=0}
// "Letter b" as a symbolic token = Parikh vector (0,1)
//           → PresburgerSet  {v | v[0]=0, v[1]=1}
//
// Helper: build_letter_set(k, letter, dim)
//   Returns the PresburgerSet for the single Parikh vector e_letter in Z^dim.

#include "symbolic_equiv.h"
#include "semilinear.h"
#include "parikh_mlir.h"
#include "nfa.h"
#include "regex.h"
#include "factorization.h"
#include <cassert>
#include <iostream>
#include <string>

using namespace mlir::presburger;

// ── Helpers ───────────────────────────────────────────────────────────────────

// Build PresburgerSet for the single vector with 1 at `letter` and 0 elsewhere.
static PresburgerSet letter_set(int letter, int dim) {
    Vec base(dim, 0);
    base[letter] = 1;
    SemilinearSet sl = {LinearSet{base, {}}};
    return to_presburger_set(sl, dim);
}

// Build PresburgerSet for {v ∈ Z^dim | v[letter] >= 1, all others = 0}
// i.e. any positive count of `letter` and nothing else.
static PresburgerSet only_letter_set(int letter, int dim) {
    Vec base(dim, 0);
    base[letter] = 1;
    Vec period(dim, 0);
    period[letter] = 1;
    SemilinearSet sl = {LinearSet{base, {period}}};
    return to_presburger_set(sl, dim);
}

// Build PresburgerSet for Z^dim with no constraints (the universe).
static PresburgerSet universe_set(int dim) {
    PresburgerSpace space = PresburgerSpace::getSetSpace(dim);
    IntegerPolyhedron poly(space);
    PresburgerSet result = PresburgerSet::getEmpty(space);
    result.unionInPlace(poly);
    return result;
}

// ── Test framework ────────────────────────────────────────────────────────────

static int pass_count = 0;
static int fail_count = 0;

static void check(bool cond, const std::string& name) {
    if (cond) {
        ++pass_count;
    } else {
        ++fail_count;
        std::cerr << "FAIL: " << name << "\n";
    }
}

// ── Minterm tests ─────────────────────────────────────────────────────────────

static void test_minterms_empty_input() {
    // No input labels → no atoms (points outside all labels are never visited).
    auto ms = compute_minterms({}, 2);
    check(ms.size() == 0, "minterms_empty_input: zero atoms");
}

static void test_minterms_single_set() {
    // One label φ: the only atom is φ itself (no complement created).
    auto phi = letter_set(0, 2);  // {(1,0)}
    auto ms = compute_minterms({phi}, 2);
    check(ms.size() == 1, "minterms_single_set: one atom");
    if (!ms.empty()) check(!ms[0].isIntegerEmpty(), "minterms_single_set: atom non-empty");
}

static void test_minterms_two_disjoint_sets() {
    // φ_a = {(1,0)}, φ_b = {(0,1)} are disjoint.
    // Atoms: φ_a, φ_b  (no "neither" atom — outside all labels not tracked).
    auto phi_a = letter_set(0, 2);
    auto phi_b = letter_set(1, 2);
    auto ms = compute_minterms({phi_a, phi_b}, 2);
    check(ms.size() == 2, "minterms_two_disjoint: two atoms");
}

static void test_minterms_identical_sets() {
    // Two identical labels: the second is entirely covered by the first atom.
    auto phi = letter_set(0, 2);
    auto ms = compute_minterms({phi, phi}, 2);
    check(ms.size() == 1, "minterms_identical_sets: one atom");
}

static void test_minterms_nested_sets() {
    // φ_1 = {(1,0),(2,0),...} = {v | v[0]>=1, v[1]=0}
    // φ_2 = {(1,0)}
    // φ_2 ⊆ φ_1.
    // After processing φ_1: atoms = {φ_1}.
    // Processing φ_2: φ_1 splits into φ_2 and φ_1\φ_2.  No new atom from phi_rest.
    // Atoms: φ_2, φ_1\φ_2  (two atoms, no complement of φ_1).
    auto phi1 = only_letter_set(0, 2);       // v[0]>=1, v[1]=0
    auto phi2 = letter_set(0, 2);            // exactly (1,0)
    auto ms = compute_minterms({phi1, phi2}, 2);
    check(ms.size() == 2, "minterms_nested: two atoms");
}

// ── check_equiv tests ─────────────────────────────────────────────────────────

static void test_equiv_both_empty() {
    // Two automata with no final states: both reject everything.
    SymbolicAutomaton a1, a2;
    a1.num_states = 1; a1.dim = 2; a1.initial = {0};
    a2.num_states = 1; a2.dim = 2; a2.initial = {0};
    check(check_equiv(a1, a2), "equiv_both_empty");
}

static void test_equiv_both_accept_epsilon() {
    // Both have a single initial+final state with no transitions.
    SymbolicAutomaton a1, a2;
    a1.num_states = 1; a1.dim = 2; a1.initial = {0}; a1.final_states = {0};
    a2.num_states = 1; a2.dim = 2; a2.initial = {0}; a2.final_states = {0};
    check(check_equiv(a1, a2), "equiv_both_accept_epsilon");
}

static void test_inequiv_epsilon_vs_empty() {
    // a1 accepts ε, a2 accepts nothing.
    SymbolicAutomaton a1, a2;
    a1.num_states = 1; a1.dim = 2; a1.initial = {0}; a1.final_states = {0};
    a2.num_states = 1; a2.dim = 2; a2.initial = {0};
    check(!check_equiv(a1, a2), "inequiv_epsilon_vs_empty");
}

static void test_equiv_single_letter_same() {
    // Both accept exactly one token: vector (1,0).
    // Structure: 0 -φ_a-> 1(final)
    int dim = 2;
    SymbolicAutomaton a1, a2;
    a1.num_states = 2; a1.dim = dim; a1.initial = {0}; a1.final_states = {1};
    a1.transitions.push_back({0, letter_set(0, dim), 1});

    a2.num_states = 2; a2.dim = dim; a2.initial = {0}; a2.final_states = {1};
    a2.transitions.push_back({0, letter_set(0, dim), 1});

    check(check_equiv(a1, a2), "equiv_single_letter_same");
}

static void test_inequiv_a_vs_b() {
    // a1 accepts one step of "a" (vector (1,0)).
    // a2 accepts one step of "b" (vector (0,1)).
    int dim = 2;
    SymbolicAutomaton a1, a2;
    a1.num_states = 2; a1.dim = dim; a1.initial = {0}; a1.final_states = {1};
    a1.transitions.push_back({0, letter_set(0, dim), 1});

    a2.num_states = 2; a2.dim = dim; a2.initial = {0}; a2.final_states = {1};
    a2.transitions.push_back({0, letter_set(1, dim), 1});

    check(!check_equiv(a1, a2), "inequiv_a_vs_b");
}

static void test_equiv_nondeterministic_choice() {
    // Both accept one step of "a" OR one step of "b".
    // a1: 0 -φ_a-> 1(final), 0 -φ_b-> 1(final)  (NFA with shared final state)
    // a2: 0 -φ_a-> 1(final), 0 -φ_b-> 2(final)  (NFA with two final states)
    int dim = 2;
    SymbolicAutomaton a1, a2;

    a1.num_states = 2; a1.dim = dim; a1.initial = {0}; a1.final_states = {1};
    a1.transitions.push_back({0, letter_set(0, dim), 1});
    a1.transitions.push_back({0, letter_set(1, dim), 1});

    a2.num_states = 3; a2.dim = dim; a2.initial = {0}; a2.final_states = {1, 2};
    a2.transitions.push_back({0, letter_set(0, dim), 1});
    a2.transitions.push_back({0, letter_set(1, dim), 2});

    check(check_equiv(a1, a2), "equiv_nondeterministic_choice");
}

static void test_equiv_self_loop() {
    // a1: 0(initial,final) -φ_a-> 0  (accepts a^* including ε)
    // a2: same structure
    int dim = 2;
    SymbolicAutomaton a1, a2;

    a1.num_states = 1; a1.dim = dim;
    a1.initial = {0}; a1.final_states = {0};
    a1.transitions.push_back({0, letter_set(0, dim), 0});

    a2.num_states = 1; a2.dim = dim;
    a2.initial = {0}; a2.final_states = {0};
    a2.transitions.push_back({0, letter_set(0, dim), 0});

    check(check_equiv(a1, a2), "equiv_self_loop");
}

static void test_inequiv_self_loop_vs_no_loop() {
    // a1 accepts a^*, a2 accepts only ε.
    int dim = 2;
    SymbolicAutomaton a1, a2;

    a1.num_states = 1; a1.dim = dim;
    a1.initial = {0}; a1.final_states = {0};
    a1.transitions.push_back({0, letter_set(0, dim), 0});

    a2.num_states = 1; a2.dim = dim;
    a2.initial = {0}; a2.final_states = {0};
    // no transitions

    check(!check_equiv(a1, a2), "inequiv_self_loop_vs_no_loop");
}

static void test_equiv_two_step_path() {
    // Both accept exactly the two-step sequence a, b.
    // a1: 0 -φ_a-> 1 -φ_b-> 2(final)
    // a2: same
    int dim = 2;
    SymbolicAutomaton a1, a2;

    a1.num_states = 3; a1.dim = dim; a1.initial = {0}; a1.final_states = {2};
    a1.transitions.push_back({0, letter_set(0, dim), 1});
    a1.transitions.push_back({1, letter_set(1, dim), 2});

    a2.num_states = 3; a2.dim = dim; a2.initial = {0}; a2.final_states = {2};
    a2.transitions.push_back({0, letter_set(0, dim), 1});
    a2.transitions.push_back({1, letter_set(1, dim), 2});

    check(check_equiv(a1, a2), "equiv_two_step_path");
}

static void test_inequiv_ab_vs_ba() {
    // a1 accepts a, b (sequence of two steps)
    // a2 accepts b, a
    int dim = 2;
    SymbolicAutomaton a1, a2;

    a1.num_states = 3; a1.dim = dim; a1.initial = {0}; a1.final_states = {2};
    a1.transitions.push_back({0, letter_set(0, dim), 1});  // a
    a1.transitions.push_back({1, letter_set(1, dim), 2});  // b

    a2.num_states = 3; a2.dim = dim; a2.initial = {0}; a2.final_states = {2};
    a2.transitions.push_back({0, letter_set(1, dim), 1});  // b
    a2.transitions.push_back({1, letter_set(0, dim), 2});  // a

    check(!check_equiv(a1, a2), "inequiv_ab_vs_ba");
}

static void test_equiv_union_ab_and_ba() {
    // a1 accepts {a,b} (a then b, OR b then a) — two paths
    // a2 same structure
    int dim = 2;
    SymbolicAutomaton a1, a2;

    // a1: 0 -a-> 1 -b-> 3(final)  and  0 -b-> 2 -a-> 3(final)
    a1.num_states = 4; a1.dim = dim; a1.initial = {0}; a1.final_states = {3};
    a1.transitions.push_back({0, letter_set(0, dim), 1});
    a1.transitions.push_back({1, letter_set(1, dim), 3});
    a1.transitions.push_back({0, letter_set(1, dim), 2});
    a1.transitions.push_back({2, letter_set(0, dim), 3});

    // a2: same (but states relabeled differently)
    a2.num_states = 4; a2.dim = dim; a2.initial = {0}; a2.final_states = {3};
    a2.transitions.push_back({0, letter_set(1, dim), 1});  // b first
    a2.transitions.push_back({1, letter_set(0, dim), 3});  // then a
    a2.transitions.push_back({0, letter_set(0, dim), 2});  // a first
    a2.transitions.push_back({2, letter_set(1, dim), 3});  // then b

    check(check_equiv(a1, a2), "equiv_union_ab_and_ba");
}

static void test_inequiv_ab_vs_union() {
    // a1 accepts only (a, b)
    // a2 accepts (a, b) and (b, a) — strictly larger language
    int dim = 2;
    SymbolicAutomaton a1, a2;

    a1.num_states = 3; a1.dim = dim; a1.initial = {0}; a1.final_states = {2};
    a1.transitions.push_back({0, letter_set(0, dim), 1});
    a1.transitions.push_back({1, letter_set(1, dim), 2});

    a2.num_states = 4; a2.dim = dim; a2.initial = {0}; a2.final_states = {3};
    a2.transitions.push_back({0, letter_set(0, dim), 1});
    a2.transitions.push_back({1, letter_set(1, dim), 3});
    a2.transitions.push_back({0, letter_set(1, dim), 2});
    a2.transitions.push_back({2, letter_set(0, dim), 3});

    check(!check_equiv(a1, a2), "inequiv_ab_vs_union");
}

static void test_equiv_universe_label() {
    // Both accept one step of anything (universe label).
    int dim = 2;
    SymbolicAutomaton a1, a2;

    a1.num_states = 2; a1.dim = dim; a1.initial = {0}; a1.final_states = {1};
    a1.transitions.push_back({0, universe_set(dim), 1});

    a2.num_states = 2; a2.dim = dim; a2.initial = {0}; a2.final_states = {1};
    a2.transitions.push_back({0, universe_set(dim), 1});

    check(check_equiv(a1, a2), "equiv_universe_label");
}

static void test_inequiv_universe_vs_letter() {
    // a1: one step of anything
    // a2: one step of "a" only
    int dim = 2;
    SymbolicAutomaton a1, a2;

    a1.num_states = 2; a1.dim = dim; a1.initial = {0}; a1.final_states = {1};
    a1.transitions.push_back({0, universe_set(dim), 1});

    a2.num_states = 2; a2.dim = dim; a2.initial = {0}; a2.final_states = {1};
    a2.transitions.push_back({0, letter_set(0, dim), 1});

    check(!check_equiv(a1, a2), "inequiv_universe_vs_letter");
}

static void test_equiv_longer_loop() {
    // Both accept (a b)^n for n >= 0.
    // a1: 0(initial,final) -a-> 1 -b-> 0
    // a2: same
    int dim = 2;
    SymbolicAutomaton a1, a2;

    a1.num_states = 2; a1.dim = dim; a1.initial = {0}; a1.final_states = {0};
    a1.transitions.push_back({0, letter_set(0, dim), 1});
    a1.transitions.push_back({1, letter_set(1, dim), 0});

    a2.num_states = 2; a2.dim = dim; a2.initial = {0}; a2.final_states = {0};
    a2.transitions.push_back({0, letter_set(0, dim), 1});
    a2.transitions.push_back({1, letter_set(1, dim), 0});

    check(check_equiv(a1, a2), "equiv_longer_loop");
}

static void test_inequiv_ab_star_vs_a_star() {
    // a1: (a b)^n, n>=0 — initial+final, 0 -a-> 1 -b-> 0
    // a2: a^n, n>=0   — initial+final, 0 -a-> 0
    int dim = 2;
    SymbolicAutomaton a1, a2;

    a1.num_states = 2; a1.dim = dim; a1.initial = {0}; a1.final_states = {0};
    a1.transitions.push_back({0, letter_set(0, dim), 1});
    a1.transitions.push_back({1, letter_set(1, dim), 0});

    a2.num_states = 1; a2.dim = dim; a2.initial = {0}; a2.final_states = {0};
    a2.transitions.push_back({0, letter_set(0, dim), 0});

    // "a" alone is accepted by a2 but not a1.
    check(!check_equiv(a1, a2), "inequiv_ab_star_vs_a_star");
}

// ── fac_to_symbolic integration test ─────────────────────────────────────────

static void test_fac_to_symbolic_basic() {
    // Build NFA for (ab)* — two classes: {a}=class0, {b}=class1.
    // Partition: 2 letters, 2 classes, letter_to_class = [0, 1].
    Partition part;
    part.num_letters = 2;
    part.num_classes = 2;
    part.letter_to_class = {0, 1};
    part.class_letters   = {{0}, {1}};

    // NFA for (ab)*: 0 -a-> 1 -b-> 0, initial={0}, final={0}.
    NFA nfa(2);
    nfa.initial      = {0};
    nfa.final_states = {0};
    nfa.add_transition(0, 0, 1);  // a
    nfa.add_transition(1, 1, 0);  // b

    FactorizationAutomaton fac = build_factorization_automaton(nfa, part);
    SymbolicAutomaton sa = fac_to_symbolic(fac, part.num_letters);

    check(sa.dim == 2,      "fac_to_symbolic_basic: dim=2");
    check(!sa.initial.empty(), "fac_to_symbolic_basic: has initial states");
    check(!sa.transitions.empty(), "fac_to_symbolic_basic: has transitions");
}

static void test_fac_to_symbolic_single_class() {
    // Single-class partition: all letters commute.
    // NFA: 0 -a-> 0 (self-loop), initial+final={0}.
    Partition part;
    part.num_letters = 1;
    part.num_classes = 1;
    part.letter_to_class = {0};
    part.class_letters   = {{0}};

    NFA nfa(1);
    nfa.initial      = {0};
    nfa.final_states = {0};
    nfa.add_transition(0, 0, 0);

    FactorizationAutomaton fac = build_factorization_automaton(nfa, part);
    SymbolicAutomaton sa = fac_to_symbolic(fac, part.num_letters);

    check(sa.dim == 1, "fac_to_symbolic_single_class: dim=1");
    // Transitions may be empty since all source and dest classes are the same.
    // (A_k1^+ transitions require k1 != k2.)
    check(sa.transitions.empty() || !sa.transitions.empty(),
          "fac_to_symbolic_single_class: runs without crash");
}

// ── parikh_image_nfa tests ────────────────────────────────────────────────────

static void test_pnfa_empty_nfa() {
    NFA nfa;  // 0 states
    auto sl = parikh_image_nfa(nfa, 2);
    check(sl.empty(), "pnfa_empty_nfa: empty");
}

static void test_pnfa_single_letter() {
    // NFA: 0 -a-> 1, initial={0}, final={1}.  Accepts {a}.
    NFA nfa(2);
    nfa.initial = {0}; nfa.final_states = {1};
    nfa.add_transition(0, 0, 1);
    auto sl = parikh_image_nfa(nfa, 2);
    check(!sl.empty(), "pnfa_single_letter: non-empty");
    check(sl_contains(sl, {1, 0}), "pnfa_single_letter: contains (1,0)");
    check(!sl_contains(sl, {0, 0}), "pnfa_single_letter: not contains (0,0)");
    check(!sl_contains(sl, {0, 1}), "pnfa_single_letter: not contains (0,1)");
    check(!sl_contains(sl, {2, 0}), "pnfa_single_letter: not contains (2,0)");
}

static void test_pnfa_accepts_epsilon() {
    // NFA: single initial+final state, no transitions.  Accepts {ε}.
    NFA nfa(1);
    nfa.initial = {0}; nfa.final_states = {0};
    auto sl = parikh_image_nfa(nfa, 2);
    check(sl_contains(sl, {0, 0}), "pnfa_accepts_epsilon: contains (0,0)");
}

static void test_pnfa_kleene_star() {
    // NFA for a*: state 0 (initial+final) with self-loop on a.
    NFA nfa(1);
    nfa.initial = {0}; nfa.final_states = {0};
    nfa.add_transition(0, 0, 0);
    auto sl = parikh_image_nfa(nfa, 2);
    check(sl_contains(sl, {0, 0}), "pnfa_kleene_star: contains (0,0)");
    check(sl_contains(sl, {1, 0}), "pnfa_kleene_star: contains (1,0)");
    check(sl_contains(sl, {3, 0}), "pnfa_kleene_star: contains (3,0)");
    check(!sl_contains(sl, {0, 1}), "pnfa_kleene_star: not contains (0,1)");
}

static void test_pnfa_two_letters() {
    // NFA: 0 -a-> 1 -b-> 2, initial={0}, final={2}.  Accepts {ab}.
    NFA nfa(3);
    nfa.initial = {0}; nfa.final_states = {2};
    nfa.add_transition(0, 0, 1);
    nfa.add_transition(1, 1, 2);
    auto sl = parikh_image_nfa(nfa, 2);
    check(sl_contains(sl, {1, 1}), "pnfa_two_letters: contains (1,1)");
    check(!sl_contains(sl, {1, 0}), "pnfa_two_letters: not contains (1,0)");
    check(!sl_contains(sl, {0, 1}), "pnfa_two_letters: not contains (0,1)");
}

static void test_pnfa_nfa_plus() {
    // NFA built by nfa_plus(make_letter_nfa(0)): accepts a^+ over 2-letter alphabet.
    NFA base = make_letter_nfa(0);
    NFA nfa  = nfa_plus(base);
    auto sl = parikh_image_nfa(nfa, 2);
    check(!sl_contains(sl, {0, 0}), "pnfa_plus: no epsilon");
    check(sl_contains(sl, {1, 0}), "pnfa_plus: contains a");
    check(sl_contains(sl, {2, 0}), "pnfa_plus: contains aa");
    check(sl_contains(sl, {5, 0}), "pnfa_plus: contains aaaaa");
    check(!sl_contains(sl, {0, 1}), "pnfa_plus: no b");
}

// ── End-to-end pipeline tests ─────────────────────────────────────────────────
//
// The full pipeline is:
//   regex → NFA (Antimirov) → FactorizationAutomaton → SymbolicAutomaton
//   → check_equiv
//
// The key theorem: KA+C ⊨ e1=e2  iff  check_equiv(sa(e1), sa(e2)).
//
// When letters a,b are in the SAME commutativity class,
//   (ab)*  and  (ba)*  must be equivalent.
//
// When a,b are in DIFFERENT classes (no commutativity),
//   (ab)*  and  (ba)*  must be inequivalent.
//
// Note on m=1 (all letters in one class):
//   With only one class, i1≠i2 is never satisfied, so the factorization
//   automaton has no transitions and degenerates to a nullability check.
//   A meaningful test therefore uses m=2: letters {a,b} share class_0
//   while a third letter c is in class_1.  The NFA for (ab)* has no c
//   transitions, so A_1^+ is empty, but the factorization automaton still
//   correctly distinguishes different Parikh images via A_0^+ transitions
//   from class_1 states to class_0 states.

// Build the factorization automaton and symbolic automaton for regex e
// under the given partition.
static SymbolicAutomaton make_sym(const RegexPtr& e, const Partition& part) {
    NFA nfa = regex_to_nfa(e, part.num_letters);
    FactorizationAutomaton fac = build_factorization_automaton(nfa, part);
    return fac_to_symbolic(fac, part.num_letters);
}

static void test_pipeline_ab_eq_ba_same_class() {
    // Σ = {a=0, b=1, c=2},  class_0={a,b},  class_1={c},  m=2.
    // (ab)* and (ba)* must be equivalent: a and b commute.
    Partition part;
    part.num_letters    = 3;
    part.num_classes    = 2;
    part.letter_to_class = {0, 0, 1};   // a→0, b→0, c→1
    part.class_letters   = {{0, 1}, {2}};

    SymbolicAutomaton sa1 = make_sym(
        re_star(re_concat(re_letter(0), re_letter(1))), part);  // (ab)*
    SymbolicAutomaton sa2 = make_sym(
        re_star(re_concat(re_letter(1), re_letter(0))), part);  // (ba)*

    check(check_equiv(sa1, sa2),
          "pipeline: (ab)*=(ba)* when a,b same class");
}

static void test_pipeline_ab_neq_ba_diff_class() {
    // Σ = {a=0, b=1},  class_0={a},  class_1={b},  m=2.
    // (ab)* and (ba)* must be INequivalent: a and b do not commute.
    Partition part;
    part.num_letters    = 2;
    part.num_classes    = 2;
    part.letter_to_class = {0, 1};      // a→0, b→1
    part.class_letters   = {{0}, {1}};

    SymbolicAutomaton sa1 = make_sym(
        re_star(re_concat(re_letter(0), re_letter(1))), part);  // (ab)*
    SymbolicAutomaton sa2 = make_sym(
        re_star(re_concat(re_letter(1), re_letter(0))), part);  // (ba)*

    check(!check_equiv(sa1, sa2),
          "pipeline: (ab)*≠(ba)* when a,b different classes");
}

static void test_pipeline_a_star_eq_a_star_same_class() {
    // Sanity: a* = a* trivially, regardless of partition.
    Partition part;
    part.num_letters    = 2;
    part.num_classes    = 2;
    part.letter_to_class = {0, 1};
    part.class_letters   = {{0}, {1}};

    SymbolicAutomaton sa1 = make_sym(re_star(re_letter(0)), part);
    SymbolicAutomaton sa2 = make_sym(re_star(re_letter(0)), part);
    check(check_equiv(sa1, sa2), "pipeline: a*=a*");
}

static void test_pipeline_a_star_neq_b_star_diff_class() {
    // a* ≠ b* when a,b are in different classes.
    Partition part;
    part.num_letters    = 2;
    part.num_classes    = 2;
    part.letter_to_class = {0, 1};
    part.class_letters   = {{0}, {1}};

    SymbolicAutomaton sa1 = make_sym(re_star(re_letter(0)), part);  // a*
    SymbolicAutomaton sa2 = make_sym(re_star(re_letter(1)), part);  // b*
    check(!check_equiv(sa1, sa2), "pipeline: a*≠b* when diff classes");
}

static void test_pipeline_epsilon_neq_ab_star_plus() {
    // ε ≠ (ab)+ even with a,b in same class  (different nullability).
    Partition part;
    part.num_letters    = 3;
    part.num_classes    = 2;
    part.letter_to_class = {0, 0, 1};
    part.class_letters   = {{0, 1}, {2}};

    SymbolicAutomaton sa1 = make_sym(re_one(), part);                            // ε
    SymbolicAutomaton sa2 = make_sym(
        re_plus(re_concat(re_letter(0), re_letter(1))), part);  // (ab)+
    check(!check_equiv(sa1, sa2),
          "pipeline: ε≠(ab)+ even with a,b same class");
}

static void test_pipeline_ab_plus_eq_ba_plus_same_class() {
    // (ab)+ = (ba)+ when a,b commute  (both non-nullable, same Parikh image).
    Partition part;
    part.num_letters    = 3;
    part.num_classes    = 2;
    part.letter_to_class = {0, 0, 1};
    part.class_letters   = {{0, 1}, {2}};

    SymbolicAutomaton sa1 = make_sym(
        re_plus(re_concat(re_letter(0), re_letter(1))), part);  // (ab)+
    SymbolicAutomaton sa2 = make_sym(
        re_plus(re_concat(re_letter(1), re_letter(0))), part);  // (ba)+
    check(check_equiv(sa1, sa2),
          "pipeline: (ab)+=(ba)+ when a,b same class");
}

// ── Shared partition helpers ──────────────────────────────────────────────────

// Σ={a=0,b=1,c=2,d=3}, class_0={a,b,c}, class_1={d}, m=2.
// Use this when a,b,c should all commute; d is a dummy letter needed to make m=2.
static Partition part_abc_same() {
    Partition p;
    p.num_letters    = 4;
    p.num_classes    = 2;
    p.letter_to_class = {0, 0, 0, 1};  // a→0, b→0, c→0, d→1
    p.class_letters   = {{0, 1, 2}, {3}};
    return p;
}

// Σ={a=0,b=1,c=2}, class_0={a}, class_1={b}, class_2={c}, m=3.
static Partition part_abc_all_diff() {
    Partition p;
    p.num_letters    = 3;
    p.num_classes    = 3;
    p.letter_to_class = {0, 1, 2};
    p.class_letters   = {{0}, {1}, {2}};
    return p;
}

// ── Word acceptance tests ─────────────────────────────────────────────────────
//
// Helpers: encode a string using 'a'=0, 'b'=1, 'c'=2, 'd'=3.

static std::vector<int> w(const std::string& s) {
    std::vector<int> v;
    for (char c : s) v.push_back(c - 'a');
    return v;
}

static void test_word_acceptance() {
    std::printf("-- word_to_blocks and sym_accepts --\n");

    // ── Partition: Σ={a,b,c}, class_0={a,b}, class_1={c}, m=2 ────────────────
    Partition p2;
    p2.num_letters    = 3;
    p2.num_classes    = 2;
    p2.letter_to_class = {0, 0, 1};   // a→0, b→0, c→1
    p2.class_letters   = {{0, 1}, {2}};

    // word_to_blocks sanity checks
    {
        // "abcba" → (ab)(c)(ba) → [(1,1,0),(0,0,1),(1,1,0)]
        auto blocks = word_to_blocks(w("abcba"), p2);
        check(blocks.size() == 3, "blocks: abcba has 3 blocks");
        if (blocks.size() == 3) {
            check(blocks[0] == Vec({1,1,0}), "blocks: block0 = (1,1,0)");
            check(blocks[1] == Vec({0,0,1}), "blocks: block1 = (0,0,1)");
            check(blocks[2] == Vec({1,1,0}), "blocks: block2 = (1,1,0)");
        }
        // "aabb" → one block (all class_0) → [(2,2,0)]
        auto blocks2 = word_to_blocks(w("aabb"), p2);
        check(blocks2.size() == 1, "blocks: aabb has 1 block");
        if (!blocks2.empty()) check(blocks2[0] == Vec({2,2,0}), "blocks: aabb block = (2,2,0)");
        // "cab" → (c)(ab) → [(0,0,1),(1,1,0)]
        auto blocks3 = word_to_blocks(w("cab"), p2);
        check(blocks3.size() == 2, "blocks: cab has 2 blocks");
        // empty word
        auto blocks4 = word_to_blocks({}, p2);
        check(blocks4.empty(), "blocks: empty word has 0 blocks");
    }

    // ── (ab)* with class_0={a,b}, class_1={c} ────────────────────────────────
    //   Accepts C-words whose blocks match (ab)+ Parikh image between class changes.
    {
        RegexPtr ab_star = re_star(re_concat(re_letter(0), re_letter(1)));

        // ε is accepted
        check(katc_accepts(ab_star, p2, {}),           "(ab)*: accepts ε");
        // "ab" → one block (1,1,0) ∈ P((ab)+) → accept
        check(katc_accepts(ab_star, p2, w("ab")),      "(ab)*: accepts 'ab'");
        // "ba" → one block (1,1,0) same as "ab" → accept (a,b commute)
        check(katc_accepts(ab_star, p2, w("ba")),      "(ab)*: accepts 'ba' (comm.)");
        // "aabb" → one block (2,2,0) ∈ P((ab)+) → accept
        check(katc_accepts(ab_star, p2, w("aabb")),    "(ab)*: accepts 'aabb' (comm.)");
        // "abab" → one block (2,2,0) → accept
        check(katc_accepts(ab_star, p2, w("abab")),    "(ab)*: accepts 'abab'");
        // "a" → block (1,0,0): a-count≠b-count → reject
        check(!katc_accepts(ab_star, p2, w("a")),      "(ab)*: rejects 'a'");
        // "c" → block (0,0,1): class_1 letter alone → reject
        check(!katc_accepts(ab_star, p2, w("c")),      "(ab)*: rejects 'c'");
        // "abc" → blocks [(1,1,0),(0,0,1)]: c block leads to dead state → reject
        check(!katc_accepts(ab_star, p2, w("abc")),    "(ab)*: rejects 'abc'");
        // "b" → block (0,1,0): only b → reject
        check(!katc_accepts(ab_star, p2, w("b")),      "(ab)*: rejects 'b'");
    }

    // ── (ab)* with class_0={a}, class_1={b} (different classes) ──────────────
    //   Now a,b do NOT commute; "ba" is different from "ab".
    {
        Partition diff;
        diff.num_letters    = 2;
        diff.num_classes    = 2;
        diff.letter_to_class = {0, 1};
        diff.class_letters   = {{0}, {1}};

        RegexPtr ab_star = re_star(re_concat(re_letter(0), re_letter(1)));

        check(katc_accepts(ab_star, diff, {}),          "(ab)* diff: accepts ε");
        check(katc_accepts(ab_star, diff, w("ab")),     "(ab)* diff: accepts 'ab'");
        check(katc_accepts(ab_star, diff, w("abab")),   "(ab)* diff: accepts 'abab'");
        // "ba" → blocks [(1,0),(0,1)] but wrong order for (ab)*: reject
        check(!katc_accepts(ab_star, diff, w("ba")),    "(ab)* diff: rejects 'ba'");
        // "aabb" → blocks [(2,0),(0,2)]: class_0 block (2,0) not in P(a+) then
        //   class_1 block (0,2) → reject (a block must be size 1 per (ab)+)
        check(!katc_accepts(ab_star, diff, w("aabb")),  "(ab)* diff: rejects 'aabb'");
    }

    // ── (abc)* with class_0={a,b,c}, class_1={d} (m=2, d is dummy) ──────────
    //   All of a,b,c commute; words in (abc)* require equal counts of a,b,c.
    {
        Partition p3 = part_abc_same();  // defined earlier in this file
        RegexPtr abc_star = re_star(
            re_concat(re_letter(0), re_concat(re_letter(1), re_letter(2))));

        check(katc_accepts(abc_star, p3, {}),              "(abc)*: accepts ε");
        check(katc_accepts(abc_star, p3, w("abc")),        "(abc)*: accepts 'abc'");
        // "bca" → one block (1,1,1,0) = same Parikh as "abc" → accept
        check(katc_accepts(abc_star, p3, w("bca")),        "(abc)*: accepts 'bca' (comm.)");
        // "cba" → (1,1,1,0) → accept
        check(katc_accepts(abc_star, p3, w("cba")),        "(abc)*: accepts 'cba' (comm.)");
        // "abcabc" → one block (2,2,2,0) → accept
        check(katc_accepts(abc_star, p3, w("abcabc")),     "(abc)*: accepts 'abcabc'");
        // "abccba" → one block (2,2,2,0) → accept (6 letters, equal counts)
        check(katc_accepts(abc_star, p3, w("abccba")),     "(abc)*: accepts 'abccba' (comm.)");
        // "ab" → block (1,1,0,0): c-count=0, a=b≠c → reject
        check(!katc_accepts(abc_star, p3, w("ab")),        "(abc)*: rejects 'ab'");
        // "abcba" → one block (2,2,1,0): a=b=2, c=1 unequal → reject
        check(!katc_accepts(abc_star, p3, w("abcba")),     "(abc)*: rejects 'abcba'");
        // "d" → class_1 block (0,0,0,1): no d-transitions in (abc)* → reject
        check(!katc_accepts(abc_star, p3, w("d")),         "(abc)*: rejects 'd'");
    }

    // ── (ab+c)* with class_0={a,b}, class_1={c} ──────────────────────────────
    //   Each block is one or more a/b letters (any mix), separated by c blocks.
    //   A block (m,n,0) is accepted iff m+n >= 1 (any non-empty a/b word).
    {
        // (a+b)+ = one or more of {a,b}
        RegexPtr r = re_star(
            re_union(re_letter(0),         // a
                re_union(re_letter(1),     // b
                         re_letter(2))));  // c

        // ε accepted
        check(katc_accepts(r, p2, {}),             "(a+b+c)*: accepts ε");
        // single letters
        check(katc_accepts(r, p2, w("a")),         "(a+b+c)*: accepts 'a'");
        check(katc_accepts(r, p2, w("b")),         "(a+b+c)*: accepts 'b'");
        check(katc_accepts(r, p2, w("c")),         "(a+b+c)*: accepts 'c'");
        // multi-letter words with mixed blocks
        check(katc_accepts(r, p2, w("abcba")),     "(a+b+c)*: accepts 'abcba'");
        check(katc_accepts(r, p2, w("cac")),       "(a+b+c)*: accepts 'cac'");
        check(katc_accepts(r, p2, w("abc")),       "(a+b+c)*: accepts 'abc'");
    }

    // ── bc(cba)*a + ε with class_0={a,b,c}, class_1={d} ─────────────────────
    //   Equivalent to (abc)* under full {a,b,c} commutativity (test 1 above).
    {
        Partition p3 = part_abc_same();
        RegexPtr cba  = re_concat(re_letter(2), re_concat(re_letter(1), re_letter(0)));
        RegexPtr rhs  = re_union(
            re_concat(re_letter(1),
                re_concat(re_letter(2),
                    re_concat(re_star(cba), re_letter(0)))),
            re_one());

        check(katc_accepts(rhs, p3, {}),            "bc(cba)*a+ε: accepts ε");
        check(katc_accepts(rhs, p3, w("abc")),      "bc(cba)*a+ε: accepts 'abc'");
        check(katc_accepts(rhs, p3, w("bca")),      "bc(cba)*a+ε: accepts 'bca' (comm.)");
        check(katc_accepts(rhs, p3, w("abcabc")),   "bc(cba)*a+ε: accepts 'abcabc'");
        check(!katc_accepts(rhs, p3, w("ab")),      "bc(cba)*a+ε: rejects 'ab'");
        check(!katc_accepts(rhs, p3, w("abcba")),   "bc(cba)*a+ε: rejects 'abcba'");
    }
}

// ── New pipeline tests ────────────────────────────────────────────────────────

// 1. (abc)* = bc·(cba)*·a + ε  when a,b,c same class.
//    P((abc)*)          = {(n,n,n,0) | n≥0}
//    P(bc·(cba)*·a + ε) = {(0,0,0,0)} ∪ {(n,n,n,0) | n≥1} = same.
static void test_pipeline_abc_star_eq_bc_cba_star_a_eps() {
    auto part = part_abc_same();

    // (abc)*
    RegexPtr abc_star = re_star(
        re_concat(re_letter(0), re_concat(re_letter(1), re_letter(2))));

    // bc·(cba)*·a + ε
    RegexPtr cba = re_concat(re_letter(2), re_concat(re_letter(1), re_letter(0)));
    RegexPtr rhs = re_union(
        re_concat(re_letter(1),
            re_concat(re_letter(2),
                re_concat(re_star(cba), re_letter(0)))),
        re_one());

    check(check_equiv(make_sym(abc_star, part), make_sym(rhs, part)),
          "pipeline: (abc)*=bc(cba)*a+ε when a,b,c same class");
}

// 2a. (ac)*(bd)* ≠ (ab)*(cd)*  when all letters in different classes (m=4).
//     P((ac)*(bd)*) = {(n,m,n,m)} vs P((ab)*(cd)*) = {(n,n,m,m)}: different.
static void test_pipeline_acbd_neq_abcd_all_diff() {
    Partition part;
    part.num_letters    = 4;
    part.num_classes    = 4;
    part.letter_to_class = {0, 1, 2, 3};
    part.class_letters   = {{0}, {1}, {2}, {3}};

    // (ac)*(bd)*
    RegexPtr ac_bd = re_concat(
        re_star(re_concat(re_letter(0), re_letter(2))),
        re_star(re_concat(re_letter(1), re_letter(3))));

    // (ab)*(cd)*
    RegexPtr ab_cd = re_concat(
        re_star(re_concat(re_letter(0), re_letter(1))),
        re_star(re_concat(re_letter(2), re_letter(3))));

    check(!check_equiv(make_sym(ac_bd, part), make_sym(ab_cd, part)),
          "pipeline: (ac)*(bd)*≠(ab)*(cd)* when all diff classes");
}

// 2b. (ac)*(bd)* ≠ (ab)*(cd)*  when {a,c}=class_0, {b,d}=class_1.
static void test_pipeline_acbd_neq_abcd_paired_classes() {
    Partition part;
    part.num_letters    = 4;
    part.num_classes    = 2;
    part.letter_to_class = {0, 1, 0, 1};  // a→0, b→1, c→0, d→1
    part.class_letters   = {{0, 2}, {1, 3}};

    RegexPtr ac_bd = re_concat(
        re_star(re_concat(re_letter(0), re_letter(2))),
        re_star(re_concat(re_letter(1), re_letter(3))));
    RegexPtr ab_cd = re_concat(
        re_star(re_concat(re_letter(0), re_letter(1))),
        re_star(re_concat(re_letter(2), re_letter(3))));

    check(!check_equiv(make_sym(ac_bd, part), make_sym(ab_cd, part)),
          "pipeline: (ac)*(bd)*≠(ab)*(cd)* when {a,c} and {b,d} paired");
}

// 3. (ab)*(ba)* = (ab)*  when a,b same class.
//    P((ab)*(ba)*) = {(k,k) | k≥0} = P((ab)*).
static void test_pipeline_abstar_bastar_eq_abstar() {
    Partition part;
    part.num_letters    = 3;
    part.num_classes    = 2;
    part.letter_to_class = {0, 0, 1};
    part.class_letters   = {{0, 1}, {2}};

    RegexPtr ab_star = re_star(re_concat(re_letter(0), re_letter(1)));
    RegexPtr ba_star = re_star(re_concat(re_letter(1), re_letter(0)));

    check(check_equiv(
              make_sym(re_concat(ab_star, ba_star), part),
              make_sym(ab_star, part)),
          "pipeline: (ab)*(ba)*=(ab)* when a,b same class");
}

// 4. (abc)* = (bca)*  when a,b,c same class  (cyclic rotation).
static void test_pipeline_abc_star_eq_bca_star() {
    auto part = part_abc_same();

    RegexPtr abc = re_star(re_concat(re_letter(0), re_concat(re_letter(1), re_letter(2))));
    RegexPtr bca = re_star(re_concat(re_letter(1), re_concat(re_letter(2), re_letter(0))));

    check(check_equiv(make_sym(abc, part), make_sym(bca, part)),
          "pipeline: (abc)*=(bca)* when a,b,c same class");
}

// 5. (abc)* = (cba)*  when a,b,c same class  (reversal).
static void test_pipeline_abc_star_eq_cba_star() {
    auto part = part_abc_same();

    RegexPtr abc = re_star(re_concat(re_letter(0), re_concat(re_letter(1), re_letter(2))));
    RegexPtr cba = re_star(re_concat(re_letter(2), re_concat(re_letter(1), re_letter(0))));

    check(check_equiv(make_sym(abc, part), make_sym(cba, part)),
          "pipeline: (abc)*=(cba)* when a,b,c same class");
}

// 6. (abc)* = (acb)*  when a,b,c same class  (any permutation is equivalent).
static void test_pipeline_abc_star_eq_acb_star_same() {
    auto part = part_abc_same();

    RegexPtr abc = re_star(re_concat(re_letter(0), re_concat(re_letter(1), re_letter(2))));
    RegexPtr acb = re_star(re_concat(re_letter(0), re_concat(re_letter(2), re_letter(1))));

    check(check_equiv(make_sym(abc, part), make_sym(acb, part)),
          "pipeline: (abc)*=(acb)* when a,b,c same class");
}

// 7. (abc)* ≠ (acb)*  when a,b,c all in different classes (m=3).
//    Same Parikh image, but different block ordering → inequivalent.
static void test_pipeline_abc_star_neq_acb_star_diff() {
    auto part = part_abc_all_diff();

    RegexPtr abc = re_star(re_concat(re_letter(0), re_concat(re_letter(1), re_letter(2))));
    RegexPtr acb = re_star(re_concat(re_letter(0), re_concat(re_letter(2), re_letter(1))));

    check(!check_equiv(make_sym(abc, part), make_sym(acb, part)),
          "pipeline: (abc)*≠(acb)* when a,b,c all different classes");
}

// 8. (ab)* + (ba)* = (ab)*  when a,b same class.
//    Union of two expressions with the same Parikh image equals either one.
static void test_pipeline_ab_union_ba_eq_ab() {
    Partition part;
    part.num_letters    = 3;
    part.num_classes    = 2;
    part.letter_to_class = {0, 0, 1};
    part.class_letters   = {{0, 1}, {2}};

    RegexPtr ab_star = re_star(re_concat(re_letter(0), re_letter(1)));
    RegexPtr ba_star = re_star(re_concat(re_letter(1), re_letter(0)));

    check(check_equiv(
              make_sym(re_union(ab_star, ba_star), part),
              make_sym(ab_star, part)),
          "pipeline: (ab)*+(ba)*=(ab)* when a,b same class");
}

// 9. (abc)*(acb)* = (abc)*  when a,b,c same class  (generalises test 3).
//    P((abc)*(acb)*) = {(k,k,k,0) | k≥0} = P((abc)*).
static void test_pipeline_abc_acb_concat_eq_abc() {
    auto part = part_abc_same();

    RegexPtr abc = re_star(re_concat(re_letter(0), re_concat(re_letter(1), re_letter(2))));
    RegexPtr acb = re_star(re_concat(re_letter(0), re_concat(re_letter(2), re_letter(1))));

    check(check_equiv(
              make_sym(re_concat(abc, acb), part),
              make_sym(abc, part)),
          "pipeline: (abc)*(acb)*=(abc)* when a,b,c same class");
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    // parikh_image_nfa tests
    test_pnfa_empty_nfa();
    test_pnfa_single_letter();
    test_pnfa_accepts_epsilon();
    test_pnfa_kleene_star();
    test_pnfa_two_letters();
    test_pnfa_nfa_plus();

    // minterm tests
    test_minterms_empty_input();
    test_minterms_single_set();
    test_minterms_two_disjoint_sets();
    test_minterms_identical_sets();
    test_minterms_nested_sets();

    // check_equiv tests
    test_equiv_both_empty();
    test_equiv_both_accept_epsilon();
    test_inequiv_epsilon_vs_empty();
    test_equiv_single_letter_same();
    test_inequiv_a_vs_b();
    test_equiv_nondeterministic_choice();
    test_equiv_self_loop();
    test_inequiv_self_loop_vs_no_loop();
    test_equiv_two_step_path();
    test_inequiv_ab_vs_ba();
    test_equiv_union_ab_and_ba();
    test_inequiv_ab_vs_union();
    test_equiv_universe_label();
    test_inequiv_universe_vs_letter();
    test_equiv_longer_loop();
    test_inequiv_ab_star_vs_a_star();

    // fac_to_symbolic integration tests
    test_fac_to_symbolic_basic();
    test_fac_to_symbolic_single_class();

    // End-to-end pipeline tests
    test_pipeline_ab_eq_ba_same_class();
    test_pipeline_ab_neq_ba_diff_class();
    test_pipeline_a_star_eq_a_star_same_class();
    test_pipeline_a_star_neq_b_star_diff_class();
    test_pipeline_epsilon_neq_ab_star_plus();
    test_pipeline_ab_plus_eq_ba_plus_same_class();

    // Word acceptance tests
    test_word_acceptance();

    // Extended pipeline tests
    test_pipeline_abc_star_eq_bc_cba_star_a_eps();  // (abc)*=bc(cba)*a+ε, same class
    test_pipeline_acbd_neq_abcd_all_diff();          // (ac)*(bd)*≠(ab)*(cd)*, all diff
    test_pipeline_acbd_neq_abcd_paired_classes();    // (ac)*(bd)*≠(ab)*(cd)*, paired
    test_pipeline_abstar_bastar_eq_abstar();         // (ab)*(ba)*=(ab)*, same class
    test_pipeline_abc_star_eq_bca_star();            // (abc)*=(bca)*, same class
    test_pipeline_abc_star_eq_cba_star();            // (abc)*=(cba)*, same class
    test_pipeline_abc_star_eq_acb_star_same();       // (abc)*=(acb)*, same class
    test_pipeline_abc_star_neq_acb_star_diff();      // (abc)*≠(acb)*, all diff classes
    test_pipeline_ab_union_ba_eq_ab();               // (ab)*+(ba)*=(ab)*, same class
    test_pipeline_abc_acb_concat_eq_abc();           // (abc)*(acb)*=(abc)*, same class

    std::cout << pass_count << " passed, " << fail_count << " failed.\n";
    return fail_count > 0 ? 1 : 0;
}

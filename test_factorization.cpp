#include "factorization.h"
#include "regex.h"
#include <cassert>
#include <cstdio>
#include <string>

// ---- Minimal test harness ----
static int passed = 0, failed = 0;

static void check(bool cond, const char* msg) {
    if (cond) { ++passed; std::printf("  PASS: %s\n", msg); }
    else       { ++failed; std::printf("  FAIL: %s\n", msg); }
}

// ---- Helpers ----

// Encode a string as a vector<int>: 'a'=0, 'b'=1, 'c'=2, ...
static std::vector<int> word(const std::string& s) {
    std::vector<int> v;
    for (char c : s) v.push_back(c - 'a');
    return v;
}

// Convert a RegexPtr to NFA (over k letters) and test acceptance
static bool re_accepts(const RegexPtr& r, const std::vector<int>& w, int k) {
    return regex_to_nfa(r, k).accepts(w);
}

// ============================================================
// Test suite 1: Basic NFA operations (unchanged)
// ============================================================
static void test_make_letter_nfa() {
    std::printf("-- make_letter_nfa --\n");
    NFA n = make_letter_nfa(0); // accepts {a}
    check( n.accepts(word("a")),  "accepts 'a'");
    check(!n.accepts(word("b")),  "rejects 'b'");
    check(!n.accepts(word("aa")), "rejects 'aa'");
    check(!n.accepts({}),         "rejects ε");
    check(!n.is_empty(),          "not empty");
}

static void test_make_epsilon_nfa() {
    std::printf("-- make_epsilon_nfa --\n");
    NFA n = make_epsilon_nfa();
    check( n.accepts({}),         "accepts ε");
    check(!n.accepts(word("a")),  "rejects 'a'");
    check( n.accepts_epsilon(),   "accepts_epsilon true");
}

static void test_make_empty_nfa() {
    std::printf("-- make_empty_nfa --\n");
    NFA n = make_empty_nfa();
    check(!n.accepts({}),         "rejects ε");
    check(!n.accepts(word("a")),  "rejects 'a'");
    check( n.is_empty(),          "is_empty true");
}

static void test_nfa_union() {
    std::printf("-- nfa_union --\n");
    NFA u = nfa_union(make_letter_nfa(0), make_letter_nfa(1)); // {a} ∪ {b}
    check( u.accepts(word("a")),  "accepts 'a'");
    check( u.accepts(word("b")),  "accepts 'b'");
    check(!u.accepts(word("c")),  "rejects 'c'");
    check(!u.accepts({}),         "rejects ε");
    check(!u.accepts(word("ab")), "rejects 'ab'");
}

static void test_nfa_concat() {
    std::printf("-- nfa_concat --\n");
    NFA ab = nfa_concat(make_letter_nfa(0), make_letter_nfa(1));
    check( ab.accepts(word("ab")), "accepts 'ab'");
    check(!ab.accepts(word("a")),  "rejects 'a'");
    check(!ab.accepts(word("b")),  "rejects 'b'");
    check(!ab.accepts(word("ba")), "rejects 'ba'");
    check(!ab.accepts({}),         "rejects ε");

    NFA ae = nfa_concat(make_letter_nfa(0), make_epsilon_nfa());
    check( ae.accepts(word("a")),  "concat with ε accepts 'a'");
    check(!ae.accepts(word("ab")), "concat with ε rejects 'ab'");

    NFA ea = nfa_concat(make_epsilon_nfa(), make_letter_nfa(0));
    check( ea.accepts(word("a")),  "ε concat accepts 'a'");
}

static void test_nfa_star() {
    std::printf("-- nfa_star --\n");
    NFA astar = nfa_star(make_letter_nfa(0)); // a*
    check( astar.accepts({}),          "accepts ε");
    check( astar.accepts(word("a")),   "accepts 'a'");
    check( astar.accepts(word("aa")),  "accepts 'aa'");
    check( astar.accepts(word("aaa")), "accepts 'aaa'");
    check(!astar.accepts(word("b")),   "rejects 'b'");
    check(!astar.accepts(word("ab")),  "rejects 'ab'");

    NFA estar = nfa_star(make_empty_nfa());
    check( estar.accepts({}),         "(∅)* accepts ε");
    check(!estar.accepts(word("a")),  "(∅)* rejects 'a'");
}

static void test_nfa_plus() {
    std::printf("-- nfa_plus --\n");
    NFA aplus = nfa_plus(make_letter_nfa(0)); // a+
    check(!aplus.accepts({}),          "rejects ε");
    check( aplus.accepts(word("a")),   "accepts 'a'");
    check( aplus.accepts(word("aa")),  "accepts 'aa'");
    check(!aplus.accepts(word("b")),   "rejects 'b'");
}

// ============================================================
// Test suite 2: Matrix star (RegexPtr semiring)
// ============================================================

static MatSR<RegexPtr> make_re_sr() {
    return MatSR<RegexPtr>{
        re_zero(),
        [](RegexPtr a, RegexPtr b) { return re_union(a, b); },
        [](RegexPtr a, RegexPtr b) { return re_concat(a, b); },
        [](RegexPtr a)             { return re_star(a); }
    };
}

// 2-state matrix: A[0][0] = {a},  all others zero
// Expected A^*: A^*[0][0] = a*, A^*[0][1] = ∅, A^*[1][0] = ∅, A^*[1][1] = ε
static void test_matrix_star_self_loop() {
    std::printf("-- matrix_star (self-loop) --\n");
    auto sr = make_re_sr();
    int n = 2, k = 1; // alphabet size 1 (only 'a')
    NfaMatrix<RegexPtr> A = mat_zero<RegexPtr>(n, n, re_zero());
    A[0][0] = re_letter(0); // a

    auto Astar = matrix_star(A, sr);

    // A^*[0][0] = a*
    check( re_accepts(Astar[0][0], {}, k),         "A*[0][0] accepts ε");
    check( re_accepts(Astar[0][0], word("a"), k),  "A*[0][0] accepts 'a'");
    check( re_accepts(Astar[0][0], word("aa"), k), "A*[0][0] accepts 'aa'");
    check(!re_accepts(Astar[0][0], word("b"), 2),  "A*[0][0] rejects 'b'");

    // A^*[1][1] = ε
    check( re_accepts(Astar[1][1], {}, k),         "A*[1][1] accepts ε");
    check(!re_accepts(Astar[1][1], word("a"), k),  "A*[1][1] rejects 'a'");

    // A^*[0][1] = ∅
    check( Astar[0][1]->kind == Regex::Kind::Zero, "A*[0][1] is zero");

    // A^*[1][0] = ∅
    check( Astar[1][0]->kind == Regex::Kind::Zero, "A*[1][0] is zero");
}

// 2-state machine: 0 -a-> 1
// Expected A^*: A^*[0][0]=ε, A^*[0][1]=a, A^*[1][0]=∅, A^*[1][1]=ε
static void test_matrix_star_single_edge() {
    std::printf("-- matrix_star (single edge) --\n");
    auto sr = make_re_sr();
    int n = 2, k = 1;
    NfaMatrix<RegexPtr> A = mat_zero<RegexPtr>(n, n, re_zero());
    A[0][1] = re_letter(0); // 0 -a-> 1

    auto Astar = matrix_star(A, sr);

    check( re_accepts(Astar[0][0], {}, k),         "A*[0][0] accepts ε");
    check(!re_accepts(Astar[0][0], word("a"), k),  "A*[0][0] rejects 'a'");
    check( re_accepts(Astar[0][1], word("a"), k),  "A*[0][1] accepts 'a'");
    check(!re_accepts(Astar[0][1], {}, k),         "A*[0][1] rejects ε");
    check(!re_accepts(Astar[0][1], word("aa"), k), "A*[0][1] rejects 'aa'");
    check( re_accepts(Astar[1][1], {}, k),         "A*[1][1] accepts ε");
    check( Astar[1][0]->kind == Regex::Kind::Zero, "A*[1][0] is zero");
}

// 2-state cycle: 0 -a-> 1, 1 -b-> 0
// A^+[0][0] = (ab)+, A^+[0][1] = a(ba)*
static void test_matrix_plus_cycle() {
    std::printf("-- A^+ cycle test --\n");
    auto sr = make_re_sr();
    int n = 2, k = 2; // alphabet {a=0, b=1}
    NfaMatrix<RegexPtr> A = mat_zero<RegexPtr>(n, n, re_zero());
    A[0][1] = re_letter(0); // 0 -a-> 1
    A[1][0] = re_letter(1); // 1 -b-> 0

    auto Astar = matrix_star(A, sr);
    auto Aplus = mat_mul(A, Astar, sr);

    // A^+[0][0] = (ab)+
    check( re_accepts(Aplus[0][0], word("ab"), k),   "A+[0][0] accepts 'ab'");
    check( re_accepts(Aplus[0][0], word("abab"), k), "A+[0][0] accepts 'abab'");
    check(!re_accepts(Aplus[0][0], {}, k),           "A+[0][0] rejects ε");
    check(!re_accepts(Aplus[0][0], word("a"), k),    "A+[0][0] rejects 'a'");
    check(!re_accepts(Aplus[0][0], word("b"), k),    "A+[0][0] rejects 'b'");

    // A^+[0][1] = a(ba)*
    check( re_accepts(Aplus[0][1], word("a"), k),    "A+[0][1] accepts 'a'");
    check( re_accepts(Aplus[0][1], word("aba"), k),  "A+[0][1] accepts 'aba'");
    check(!re_accepts(Aplus[0][1], {}, k),           "A+[0][1] rejects ε");
    check(!re_accepts(Aplus[0][1], word("b"), k),    "A+[0][1] rejects 'b'");
}

// ============================================================
// Test suite 3: Factorization automaton — paper example
//
// Σ = {a=0, b=1, c=2}
// C = {(a,b)} → class 0 = {a,b}, class 1 = {c}
//
// NFA A (states 0=s1, 1=s2, 2=s3):
//   0 -a-> 1,  1 -b-> 1,  1 -c-> 2,  0 -c-> 2
//   initial = {0},  final = {2}
//
// Expected M_fac transitions (from paper figure, 0-indexed):
//   (s1,class1) --[ab*]--> (s2,class0)   i.e. from=encode(0,1)=1, to=encode(1,0)=2
//   (s2,class1) --[b+]-->  (s2,class0)   i.e. from=encode(1,1)=3, to=encode(1,0)=2
//   (s1,class0) --[c]-->   (s3,class1)   i.e. from=encode(0,0)=0, to=encode(2,1)=5
//   (s2,class0) --[c]-->   (s3,class1)   i.e. from=encode(1,0)=2, to=encode(2,1)=5
// ============================================================

static NFA make_paper_nfa() {
    NFA nfa(3);
    nfa.initial      = {0};
    nfa.final_states = {2};
    nfa.add_transition(0, 0, 1); // s1 -a-> s2
    nfa.add_transition(1, 1, 1); // s2 -b-> s2
    nfa.add_transition(1, 2, 2); // s2 -c-> s3
    nfa.add_transition(0, 2, 2); // s1 -c-> s3
    return nfa;
}

static Partition make_paper_partition() {
    Partition p;
    p.num_letters = 3;
    p.num_classes = 2;
    p.letter_to_class = {0, 0, 1}; // a→0, b→0, c→1
    p.class_letters   = {{0, 1}, {2}};
    return p;
}

static void test_factorization_states() {
    std::printf("-- factorization: states --\n");
    auto fac = build_factorization_automaton(make_paper_nfa(), make_paper_partition());
    check(fac.total_states() == 6,                              "total states = 6");
    check(fac.initial.count(0) && fac.initial.count(1),        "initial = {0,1}");
    check(fac.initial.size() == 2,                              "initial size = 2");
    check(fac.final_states.count(4) && fac.final_states.count(5), "final = {4,5}");
    check(fac.final_states.size() == 2,                         "final size = 2");
}

static void test_factorization_transitions() {
    std::printf("-- factorization: transition count --\n");
    auto fac = build_factorization_automaton(make_paper_nfa(), make_paper_partition());
    check(fac.transitions.size() == 4, "transition count = 4");
}

static const FacTransition* find_trans(const FactorizationAutomaton& fac,
                                        int from, int to) {
    for (auto& t : fac.transitions)
        if (t.from == from && t.to == to) return &t;
    return nullptr;
}

static void test_factorization_labels() {
    std::printf("-- factorization: labels --\n");
    auto fac = build_factorization_automaton(make_paper_nfa(), make_paper_partition());
    const int m = 2, k = 3; // 3-letter alphabet

    int enc_s1_0 = 0*m+0; // (s1, class0)
    int enc_s1_1 = 0*m+1; // (s1, class1)
    int enc_s2_0 = 1*m+0; // (s2, class0)
    int enc_s2_1 = 1*m+1; // (s2, class1)
    int enc_s3_1 = 2*m+1; // (s3, class1)

    // Transition 1: (s1,class1) --[ab*]--> (s2,class0)
    {
        const FacTransition* t = find_trans(fac, enc_s1_1, enc_s2_0);
        check(t != nullptr, "trans (s1,1)->(s2,0) exists");
        if (t) {
            check( re_accepts(t->label, word("a"),   k), "label ab*: accepts 'a'");
            check( re_accepts(t->label, word("ab"),  k), "label ab*: accepts 'ab'");
            check( re_accepts(t->label, word("abb"), k), "label ab*: accepts 'abb'");
            check(!re_accepts(t->label, word("b"),   k), "label ab*: rejects 'b'");
            check(!re_accepts(t->label, {},          k), "label ab*: rejects ε");
            check(!re_accepts(t->label, word("ba"),  k), "label ab*: rejects 'ba'");
        }
    }

    // Transition 2: (s2,class1) --[b+]--> (s2,class0)
    {
        const FacTransition* t = find_trans(fac, enc_s2_1, enc_s2_0);
        check(t != nullptr, "trans (s2,1)->(s2,0) exists");
        if (t) {
            check( re_accepts(t->label, word("b"),  k), "label b+: accepts 'b'");
            check( re_accepts(t->label, word("bb"), k), "label b+: accepts 'bb'");
            check(!re_accepts(t->label, {},         k), "label b+: rejects ε");
            check(!re_accepts(t->label, word("a"),  k), "label b+: rejects 'a'");
        }
    }

    // Transition 3: (s1,class0) --[c]--> (s3,class1)
    {
        const FacTransition* t = find_trans(fac, enc_s1_0, enc_s3_1);
        check(t != nullptr, "trans (s1,0)->(s3,1) exists");
        if (t) {
            check( re_accepts(t->label, word("c"),  k), "label c: accepts 'c'");
            check(!re_accepts(t->label, word("cc"), k), "label c: rejects 'cc'");
            check(!re_accepts(t->label, {},         k), "label c: rejects ε");
            check(!re_accepts(t->label, word("a"),  k), "label c: rejects 'a'");
        }
    }

    // Transition 4: (s2,class0) --[c]--> (s3,class1)
    {
        const FacTransition* t = find_trans(fac, enc_s2_0, enc_s3_1);
        check(t != nullptr, "trans (s2,0)->(s3,1) exists");
        if (t) {
            check( re_accepts(t->label, word("c"),  k), "label c: accepts 'c'");
            check(!re_accepts(t->label, word("cc"), k), "label c: rejects 'cc'");
        }
    }
}

// ============================================================
// Test suite 4: Factorization with 3 classes
//
// Σ = {a=0, b=1, c=2, d=3}
// Classes: {a}=0, {b,c}=1, {d}=2
//
// NFA: 0 -a-> 1 -b-> 1 -c-> 1 -d-> 2
//      initial={0}, final={2}
// ============================================================

static void test_factorization_3classes() {
    std::printf("-- factorization: 3 classes --\n");

    NFA nfa(3);
    nfa.initial      = {0};
    nfa.final_states = {2};
    nfa.add_transition(0, 0, 1); // 0 -a-> 1
    nfa.add_transition(1, 1, 1); // 1 -b-> 1
    nfa.add_transition(1, 2, 1); // 1 -c-> 1
    nfa.add_transition(1, 3, 2); // 1 -d-> 2

    Partition p;
    p.num_letters = 4;
    p.num_classes = 3;
    p.letter_to_class  = {0, 1, 1, 2}; // a→0, b→1, c→1, d→2
    p.class_letters    = {{0}, {1,2}, {3}};

    auto fac = build_factorization_automaton(nfa, p);
    const int k = 4; // alphabet size

    check(fac.total_states() == 9, "total states = 9");

    auto enc = [&](int q, int cls){ return fac.encode(q, cls); };

    // (0,class1) --[a]--> (1,class0)
    const FacTransition* t1 = find_trans(fac, enc(0,1), enc(1,0));
    check(t1 != nullptr, "3-class: (0,1)->(1,0) exists");
    if (t1) {
        check( re_accepts(t1->label, word("a"), k), "3-class: label 'a' accepts 'a'");
        check(!re_accepts(t1->label, word("b"), k), "3-class: label 'a' rejects 'b'");
    }

    // (1,class0) --[(b+c)+]--> (1,class1)
    const FacTransition* t2 = find_trans(fac, enc(1,0), enc(1,1));
    check(t2 != nullptr, "3-class: (1,0)->(1,1) exists");
    if (t2) {
        check( re_accepts(t2->label, word("b"),  k), "3-class: label (b+c)+: 'b'");
        check( re_accepts(t2->label, word("c"),  k), "3-class: label (b+c)+: 'c'");
        check( re_accepts(t2->label, word("bc"), k), "3-class: label (b+c)+: 'bc'");
        check( re_accepts(t2->label, word("cb"), k), "3-class: label (b+c)+: 'cb'");
        check(!re_accepts(t2->label, {},         k), "3-class: label (b+c)+: rejects ε");
        check(!re_accepts(t2->label, word("a"),  k), "3-class: label (b+c)+: rejects 'a'");
    }

    // (1,class1) --[d]--> (2,class2)
    const FacTransition* t3 = find_trans(fac, enc(1,1), enc(2,2));
    check(t3 != nullptr, "3-class: (1,1)->(2,2) exists");
    if (t3) {
        check( re_accepts(t3->label, word("d"),  k), "3-class: label 'd': accepts 'd'");
        check(!re_accepts(t3->label, word("dd"), k), "3-class: label 'd': rejects 'dd'");
    }
}

// ============================================================
// main
// ============================================================
int main() {
    std::printf("=== NFA operations ===\n");
    test_make_letter_nfa();
    test_make_epsilon_nfa();
    test_make_empty_nfa();
    test_nfa_union();
    test_nfa_concat();
    test_nfa_star();
    test_nfa_plus();

    std::printf("\n=== Matrix operations (RegexPtr semiring) ===\n");
    test_matrix_star_self_loop();
    test_matrix_star_single_edge();
    test_matrix_plus_cycle();

    std::printf("\n=== Factorization automaton (paper example) ===\n");
    test_factorization_states();
    test_factorization_transitions();
    test_factorization_labels();

    std::printf("\n=== Factorization automaton (3 classes) ===\n");
    test_factorization_3classes();

    std::printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}

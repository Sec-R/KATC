#include "regex.h"
#include "factorization.h"
#include <cassert>
#include <cstdio>
#include <string>

// ── Test harness ─────────────────────────────────────────────────────────────
static int passed = 0, failed = 0;
static void check(bool cond, const char* msg) {
    if (cond) { ++passed; std::printf("  PASS: %s\n", msg); }
    else       { ++failed; std::printf("  FAIL: %s\n", msg); }
}

// Encode a string like "abc" as {0,1,2}
static std::vector<int> w(const std::string& s) {
    std::vector<int> v;
    for (char c : s) v.push_back(c - 'a');
    return v;
}

// ── Suite 1: nullable ────────────────────────────────────────────────────────
static void test_nullable() {
    std::printf("-- nullable --\n");
    check(!nullable(re_zero()),                              "ν(∅)=false");
    check( nullable(re_one()),                               "ν(ε)=true");
    check(!nullable(re_letter(0)),                           "ν(a)=false");
    check(!nullable(re_union(re_letter(0), re_letter(1))),   "ν(a+b)=false");
    check( nullable(re_union(re_one(), re_letter(0))),       "ν(ε+a)=true");
    check(!nullable(re_concat(re_letter(0), re_letter(1))),  "ν(ab)=false");
    check( nullable(re_concat(re_one(), re_one())),          "ν(εε)=true");
    check( nullable(re_star(re_letter(0))),                  "ν(a*)=true");
    check(!nullable(re_plus(re_letter(0))),                  "ν(a+)=false");
}

// ── Suite 2: partial derivatives ─────────────────────────────────────────────
static void test_partial_deriv() {
    std::printf("-- partial_deriv --\n");

    // ∂_a(∅) = ∅
    check(partial_deriv(re_zero(), 0).empty(), "∂_a(∅)=∅");

    // ∂_a(ε) = ∅
    check(partial_deriv(re_one(), 0).empty(), "∂_a(ε)=∅");

    // ∂_a(a) = {ε},  ∂_b(a) = ∅
    {
        auto da = partial_deriv(re_letter(0), 0);
        check(da.size() == 1 && (*da.begin())->id == re_one()->id, "∂_a(a)={ε}");
        check(partial_deriv(re_letter(0), 1).empty(),              "∂_b(a)=∅");
    }

    // ∂_a(a+b) = {ε},  ∂_b(a+b) = {ε}
    {
        auto r = re_union(re_letter(0), re_letter(1));
        auto da = partial_deriv(r, 0);
        auto db = partial_deriv(r, 1);
        auto dc = partial_deriv(r, 2);
        check(da.size() == 1 && (*da.begin())->id == re_one()->id, "∂_a(a+b)={ε}");
        check(db.size() == 1 && (*db.begin())->id == re_one()->id, "∂_b(a+b)={ε}");
        check(dc.empty(),                                           "∂_c(a+b)=∅");
    }

    // ∂_a(ab) = {b},  ∂_b(ab) = ∅
    {
        auto r  = re_concat(re_letter(0), re_letter(1));
        auto da = partial_deriv(r, 0);
        auto db = partial_deriv(r, 1);
        check(da.size() == 1 && (*da.begin())->id == re_letter(1)->id, "∂_a(ab)={b}");
        check(db.empty(),                                               "∂_b(ab)=∅");
    }

    // ∂_a(a*) = {a*}  (self-derivative; concat simplification: ε·a* = a*)
    {
        auto r  = re_star(re_letter(0));
        auto da = partial_deriv(r, 0);
        check(da.size() == 1 && (*da.begin())->id == r->id, "∂_a(a*)={a*}");
        check(partial_deriv(r, 1).empty(),                   "∂_b(a*)=∅");
    }

    // ∂_a(εa) = {ε}  (nullable prefix: ν(ε)=true so also ∂_a(a))
    {
        auto r  = re_concat(re_one(), re_letter(0));  // simplifies to just 'a'
        auto da = partial_deriv(r, 0);
        check(da.size() == 1 && (*da.begin())->id == re_one()->id, "∂_a(εa)={ε}");
    }
}

// ── Suite 3: regex_to_nfa — individual constructs ────────────────────────────
static void test_nfa_zero() {
    std::printf("-- regex_to_nfa(∅) --\n");
    auto nfa = regex_to_nfa(re_zero(), 2);
    check(!nfa.accepts({}),      "∅: rejects ε");
    check(!nfa.accepts(w("a")),  "∅: rejects 'a'");
    check( nfa.is_empty(),       "∅: is_empty");
}

static void test_nfa_one() {
    std::printf("-- regex_to_nfa(ε) --\n");
    auto nfa = regex_to_nfa(re_one(), 2);
    check( nfa.accepts({}),      "ε: accepts ε");
    check(!nfa.accepts(w("a")),  "ε: rejects 'a'");
}

static void test_nfa_letter() {
    std::printf("-- regex_to_nfa(a) --\n");
    auto nfa = regex_to_nfa(re_letter(0), 2); // alphabet {a,b}
    check( nfa.accepts(w("a")),  "a: accepts 'a'");
    check(!nfa.accepts(w("b")),  "a: rejects 'b'");
    check(!nfa.accepts({}),      "a: rejects ε");
    check(!nfa.accepts(w("aa")), "a: rejects 'aa'");
    // Antimirov NFA for a single letter has exactly 2 states: {a, ε}
    check(nfa.num_states == 2,   "a: 2 states");
}

static void test_nfa_union() {
    std::printf("-- regex_to_nfa(a+b) --\n");
    auto nfa = regex_to_nfa(re_union(re_letter(0), re_letter(1)), 2);
    check( nfa.accepts(w("a")),  "a+b: accepts 'a'");
    check( nfa.accepts(w("b")),  "a+b: accepts 'b'");
    check(!nfa.accepts({}),      "a+b: rejects ε");
    check(!nfa.accepts(w("ab")), "a+b: rejects 'ab'");
    check(!nfa.accepts(w("c")),  "a+b: rejects 'c' (out of alphabet)");
    // {a+b, ε} — 2 states
    check(nfa.num_states == 2,   "a+b: 2 states");
}

static void test_nfa_concat() {
    std::printf("-- regex_to_nfa(ab) --\n");
    auto nfa = regex_to_nfa(re_concat(re_letter(0), re_letter(1)), 2);
    check( nfa.accepts(w("ab")), "ab: accepts 'ab'");
    check(!nfa.accepts(w("a")),  "ab: rejects 'a'");
    check(!nfa.accepts(w("b")),  "ab: rejects 'b'");
    check(!nfa.accepts(w("ba")), "ab: rejects 'ba'");
    check(!nfa.accepts({}),      "ab: rejects ε");
    // {ab, b, ε} — 3 states
    check(nfa.num_states == 3,   "ab: 3 states");
}

static void test_nfa_star() {
    std::printf("-- regex_to_nfa(a*) --\n");
    auto nfa = regex_to_nfa(re_star(re_letter(0)), 2);
    check( nfa.accepts({}),       "a*: accepts ε");
    check( nfa.accepts(w("a")),   "a*: accepts 'a'");
    check( nfa.accepts(w("aa")),  "a*: accepts 'aa'");
    check( nfa.accepts(w("aaa")), "a*: accepts 'aaa'");
    check(!nfa.accepts(w("b")),   "a*: rejects 'b'");
    check(!nfa.accepts(w("ab")),  "a*: rejects 'ab'");
    // a* has a single self-loop state — 1 state
    check(nfa.num_states == 1,    "a*: 1 state");
}

static void test_nfa_plus() {
    std::printf("-- regex_to_nfa(a+) --\n");
    auto nfa = regex_to_nfa(re_plus(re_letter(0)), 2);
    check(!nfa.accepts({}),       "a+: rejects ε");
    check( nfa.accepts(w("a")),   "a+: accepts 'a'");
    check( nfa.accepts(w("aa")),  "a+: accepts 'aa'");
    check(!nfa.accepts(w("b")),   "a+: rejects 'b'");
}

// ── Suite 4: more complex regexes ────────────────────────────────────────────

// a(b+c)*   alphabet {a=0, b=1, c=2}
static void test_nfa_a_bc_star() {
    std::printf("-- regex_to_nfa(a(b+c)*) --\n");
    auto e   = re_concat(re_letter(0),
                   re_star(re_union(re_letter(1), re_letter(2))));
    auto nfa = regex_to_nfa(e, 3);

    check( nfa.accepts(w("a")),    "a(b+c)*: accepts 'a'");
    check( nfa.accepts(w("ab")),   "a(b+c)*: accepts 'ab'");
    check( nfa.accepts(w("ac")),   "a(b+c)*: accepts 'ac'");
    check( nfa.accepts(w("abc")),  "a(b+c)*: accepts 'abc'");
    check( nfa.accepts(w("acb")),  "a(b+c)*: accepts 'acb'");
    check( nfa.accepts(w("abbc")), "a(b+c)*: accepts 'abbc'");
    check(!nfa.accepts({}),         "a(b+c)*: rejects ε");
    check(!nfa.accepts(w("b")),    "a(b+c)*: rejects 'b'");
    check(!nfa.accepts(w("c")),    "a(b+c)*: rejects 'c'");
    check(!nfa.accepts(w("ba")),   "a(b+c)*: rejects 'ba'");
    // States: {a(b+c)*, (b+c)*} — 2 states
    // (b+c)* is nullable so it is itself the accept state; no separate ε state needed
    check(nfa.num_states == 2,     "a(b+c)*: 2 states");
}

// (a+b)*  — should accept any string over {a,b}
static void test_nfa_aorb_star() {
    std::printf("-- regex_to_nfa((a+b)*) --\n");
    auto e   = re_star(re_union(re_letter(0), re_letter(1)));
    auto nfa = regex_to_nfa(e, 2);

    check( nfa.accepts({}),         "(a+b)*: accepts ε");
    check( nfa.accepts(w("a")),     "(a+b)*: accepts 'a'");
    check( nfa.accepts(w("b")),     "(a+b)*: accepts 'b'");
    check( nfa.accepts(w("ab")),    "(a+b)*: accepts 'ab'");
    check( nfa.accepts(w("ba")),    "(a+b)*: accepts 'ba'");
    check( nfa.accepts(w("aabb")),  "(a+b)*: accepts 'aabb'");
    // (a+b)* self-loops: only 1 state
    check(nfa.num_states == 1,      "(a+b)*: 1 state");
}

// a(b+c)*d  alphabet {a=0,b=1,c=2,d=3}
static void test_nfa_abcd() {
    std::printf("-- regex_to_nfa(a(b+c)*d) --\n");
    auto bc_star = re_star(re_union(re_letter(1), re_letter(2)));
    auto e       = re_concat(re_letter(0), re_concat(bc_star, re_letter(3)));
    auto nfa     = regex_to_nfa(e, 4);

    check( nfa.accepts(w("ad")),    "a(b+c)*d: accepts 'ad'");
    check( nfa.accepts(w("abd")),   "a(b+c)*d: accepts 'abd'");
    check( nfa.accepts(w("acd")),   "a(b+c)*d: accepts 'acd'");
    check( nfa.accepts(w("abcd")),  "a(b+c)*d: accepts 'abcd'");
    check( nfa.accepts(w("abcbd")), "a(b+c)*d: accepts 'abcbd'");
    check(!nfa.accepts({}),          "a(b+c)*d: rejects ε");
    check(!nfa.accepts(w("a")),     "a(b+c)*d: rejects 'a'");
    check(!nfa.accepts(w("d")),     "a(b+c)*d: rejects 'd'");
    check(!nfa.accepts(w("ab")),    "a(b+c)*d: rejects 'ab'");
    check(!nfa.accepts(w("da")),    "a(b+c)*d: rejects 'da'");
    // States: {a(b+c)*d, (b+c)*d, ε} — 3 states
    check(nfa.num_states == 3,      "a(b+c)*d: 3 states");
}

// ── Suite 5: factory simplification rules ────────────────────────────────────
static void test_simplifications() {
    std::printf("-- factory simplifications --\n");
    // 0 + e = e
    check(re_union(re_zero(), re_letter(0))->id == re_letter(0)->id, "0+a = a");
    // e + 0 = e
    check(re_union(re_letter(0), re_zero())->id == re_letter(0)->id, "a+0 = a");
    // e + e = e
    check(re_union(re_letter(0), re_letter(0))->id == re_letter(0)->id, "a+a = a");
    // 0 · e = 0
    check(re_concat(re_zero(), re_letter(0))->id == re_zero()->id, "0·a = 0");
    // e · 0 = 0
    check(re_concat(re_letter(0), re_zero())->id == re_zero()->id, "a·0 = 0");
    // ε · e = e
    check(re_concat(re_one(), re_letter(0))->id == re_letter(0)->id, "ε·a = a");
    // e · ε = e
    check(re_concat(re_letter(0), re_one())->id == re_letter(0)->id, "a·ε = a");
    // 0* = ε
    check(re_star(re_zero())->id == re_one()->id, "0* = ε");
    // ε* = ε
    check(re_star(re_one())->id == re_one()->id,  "ε* = ε");
    // (r*)* = r*
    auto astar = re_star(re_letter(0));
    check(re_star(astar)->id == astar->id, "(a*)* = a*");
    // commutativity: a+b = b+a (same id)
    check(re_union(re_letter(0), re_letter(1))->id ==
          re_union(re_letter(1), re_letter(0))->id, "a+b = b+a (canonical)");
}

// ── Suite 6: regex_to_nfa → build_factorization_automaton ────────────────────
//
// Reproduce the paper example NFA via regex instead of manual construction:
//   regex: a·b*·c + c   (= language of the paper NFA)
//   Σ = {a=0, b=1, c=2},  C: class0={a,b}, class1={c}
static void test_regex_then_factorization() {
    std::printf("-- regex → NFA → factorization (paper example) --\n");

    // a·b*·c  +  c
    auto abc = re_concat(re_letter(0),
                  re_concat(re_star(re_letter(1)), re_letter(2)));
    auto e   = re_union(abc, re_letter(2));
    auto nfa = regex_to_nfa(e, 3);

    // NFA must accept exactly: c, ab*c
    check( nfa.accepts(w("c")),    "NFA accepts 'c'");
    check( nfa.accepts(w("ac")),   "NFA accepts 'ac'");
    check( nfa.accepts(w("abc")),  "NFA accepts 'abc'");
    check( nfa.accepts(w("abbc")), "NFA accepts 'abbc'");
    check(!nfa.accepts({}),         "NFA rejects ε");
    check(!nfa.accepts(w("a")),    "NFA rejects 'a'");
    check(!nfa.accepts(w("b")),    "NFA rejects 'b'");
    check(!nfa.accepts(w("bc")),   "NFA rejects 'bc'");
    check(!nfa.accepts(w("cc")),   "NFA rejects 'cc'");

    // Build factorization automaton
    Partition part;
    part.num_letters    = 3;
    part.num_classes    = 2;
    part.letter_to_class  = {0, 0, 1}; // a→0, b→0, c→1
    part.class_letters    = {{0, 1}, {2}};

    auto fac = build_factorization_automaton(nfa, part);

    // Factorization should have the same structure: initial×2 classes, final×2 classes
    check(fac.initial.size()     == nfa.initial.size()      * 2, "fac initial size");
    check(fac.final_states.size() == nfa.final_states.size() * 2, "fac final size");
    check(!fac.transitions.empty(), "fac has transitions");
}

// ── main ─────────────────────────────────────────────────────────────────────
int main() {
    std::printf("=== nullable ===\n");
    test_nullable();

    std::printf("\n=== partial_deriv ===\n");
    test_partial_deriv();

    std::printf("\n=== regex_to_nfa: basic constructs ===\n");
    test_nfa_zero();
    test_nfa_one();
    test_nfa_letter();
    test_nfa_union();
    test_nfa_concat();
    test_nfa_star();
    test_nfa_plus();

    std::printf("\n=== regex_to_nfa: complex regexes ===\n");
    test_nfa_a_bc_star();
    test_nfa_aorb_star();
    test_nfa_abcd();

    std::printf("\n=== factory simplifications ===\n");
    test_simplifications();

    std::printf("\n=== regex → NFA → factorization ===\n");
    test_regex_then_factorization();

    std::printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}

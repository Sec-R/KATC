#include "semilinear.h"
#include "regex.h"
#include <cassert>
#include <cstdio>
#include <string>

// ── Test harness ──────────────────────────────────────────────────────────────
static int passed = 0, failed = 0;
static void check(bool cond, const char* msg) {
    if (cond) { ++passed; std::printf("  PASS: %s\n", msg); }
    else       { ++failed; std::printf("  FAIL: %s\n", msg); }
}
static Vec v(std::initializer_list<int64_t> il) { return Vec(il); }

// ── Suite 1: base cases ───────────────────────────────────────────────────────

static void test_zero() {
    std::printf("-- P(0) --\n");
    auto S = parikh_image(re_zero(), 2);
    check(S.empty(),               "P(0) is empty");
    check(!sl_contains(S, v({0,0})), "P(0) does not contain (0,0)");
    check(!sl_contains(S, v({1,0})), "P(0) does not contain (1,0)");
}

static void test_epsilon() {
    std::printf("-- P(eps) --\n");
    auto S = parikh_image(re_one(), 2);
    check(S.size() == 1,             "P(eps) has 1 linear set");
    check(S[0].periods.empty(),      "P(eps) has no periods");
    check( sl_contains(S, v({0,0})), "P(eps) contains (0,0)");
    check(!sl_contains(S, v({1,0})), "P(eps) does not contain (1,0)");
    check(!sl_contains(S, v({0,1})), "P(eps) does not contain (0,1)");
}

static void test_letter() {
    std::printf("-- P(a), P(b) --\n");
    // Alphabet {a=0, b=1}
    auto Sa = parikh_image(re_letter(0), 2);
    check( sl_contains(Sa, v({1,0})), "P(a) contains (1,0)");
    check(!sl_contains(Sa, v({0,0})), "P(a) does not contain (0,0)");
    check(!sl_contains(Sa, v({0,1})), "P(a) does not contain (0,1)");
    check(!sl_contains(Sa, v({2,0})), "P(a) does not contain (2,0)");

    auto Sb = parikh_image(re_letter(1), 2);
    check( sl_contains(Sb, v({0,1})), "P(b) contains (0,1)");
    check(!sl_contains(Sb, v({1,0})), "P(b) does not contain (1,0)");
}

// ── Suite 2: union and concat ─────────────────────────────────────────────────

static void test_union() {
    std::printf("-- P(a+b) --\n");
    // P(a+b) = {(1,0), (0,1)}
    auto e = re_union(re_letter(0), re_letter(1));
    auto S = parikh_image(e, 2);
    check(S.size() == 2,              "P(a+b) has 2 linear sets");
    check( sl_contains(S, v({1,0})),  "P(a+b) contains (1,0)");
    check( sl_contains(S, v({0,1})),  "P(a+b) contains (0,1)");
    check(!sl_contains(S, v({0,0})),  "P(a+b) does not contain (0,0)");
    check(!sl_contains(S, v({1,1})),  "P(a+b) does not contain (1,1)");
    check(!sl_contains(S, v({2,0})),  "P(a+b) does not contain (2,0)");
}

static void test_concat() {
    std::printf("-- P(ab) --\n");
    // P(ab) = P(a) Mink P(b) = {(1,0)} + {(0,1)} = {(1,1)}
    auto e = re_concat(re_letter(0), re_letter(1));
    auto S = parikh_image(e, 2);
    check( sl_contains(S, v({1,1})),  "P(ab) contains (1,1)");
    check(!sl_contains(S, v({0,0})),  "P(ab) does not contain (0,0)");
    check(!sl_contains(S, v({1,0})),  "P(ab) does not contain (1,0)");
    check(!sl_contains(S, v({2,1})),  "P(ab) does not contain (2,1)");
}

// ── Suite 3: star ─────────────────────────────────────────────────────────────

static void test_star_letter() {
    std::printf("-- P(a*) --\n");
    // P(a*) = { n*(1,0) | n in N }  = 0 + <(1,0)>
    auto S = parikh_image(re_star(re_letter(0)), 2);
    check(S.size() == 1,              "P(a*) has 1 linear set");
    check(S[0].periods.size() == 1,   "P(a*) has 1 period");
    check( sl_contains(S, v({0,0})),  "P(a*) contains (0,0)");
    check( sl_contains(S, v({1,0})),  "P(a*) contains (1,0)");
    check( sl_contains(S, v({5,0})),  "P(a*) contains (5,0)");
    check(!sl_contains(S, v({0,1})),  "P(a*) does not contain (0,1)");
    check(!sl_contains(S, v({1,1})),  "P(a*) does not contain (1,1)");
}

static void test_star_empty() {
    std::printf("-- P(0*) --\n");
    // P(0*) = {0^k}  since (∅)* = {eps}
    auto S = parikh_image(re_star(re_zero()), 2);
    check(S.size() == 1,              "P(0*) has 1 linear set");
    check(S[0].periods.empty(),       "P(0*) has no periods");
    check( sl_contains(S, v({0,0})),  "P(0*) contains (0,0)");
    check(!sl_contains(S, v({1,0})),  "P(0*) does not contain (1,0)");
}

static void test_star_union() {
    std::printf("-- P((a+b)*) --\n");
    // P(a+b) = {(1,0),(0,1)},  P((a+b)*) = 0 + <(1,0),(0,1)> = N^2
    auto e = re_star(re_union(re_letter(0), re_letter(1)));
    auto S = parikh_image(e, 2);
    check(S.size() == 1,              "P((a+b)*) has 1 linear set");
    check(S[0].periods.size() == 2,   "P((a+b)*) has 2 periods");
    // Should contain every (x,y) in N^2
    check( sl_contains(S, v({0,0})),  "P((a+b)*) contains (0,0)");
    check( sl_contains(S, v({1,0})),  "P((a+b)*) contains (1,0)");
    check( sl_contains(S, v({0,1})),  "P((a+b)*) contains (0,1)");
    check( sl_contains(S, v({3,5})),  "P((a+b)*) contains (3,5)");
    check( sl_contains(S, v({7,7})),  "P((a+b)*) contains (7,7)");
}

static void test_star_concat() {
    std::printf("-- P((ab)*) --\n");
    // P(ab) = {(1,1)},  P((ab)*) = 0 + <(1,1)>  = {n*(1,1) | n in N}
    auto e = re_star(re_concat(re_letter(0), re_letter(1)));
    auto S = parikh_image(e, 2);
    check(S.size() == 1,              "P((ab)*) has 1 linear set");
    check(S[0].periods.size() == 1,   "P((ab)*) has 1 period");
    check( sl_contains(S, v({0,0})),  "P((ab)*) contains (0,0)");
    check( sl_contains(S, v({1,1})),  "P((ab)*) contains (1,1)");
    check( sl_contains(S, v({4,4})),  "P((ab)*) contains (4,4)");
    check(!sl_contains(S, v({1,0})),  "P((ab)*) does not contain (1,0)");
    check(!sl_contains(S, v({0,1})),  "P((ab)*) does not contain (0,1)");
    check(!sl_contains(S, v({2,3})),  "P((ab)*) does not contain (2,3)");
}

// ── Suite 4: paper examples ───────────────────────────────────────────────────

static void test_paper_ex1() {
    std::printf("-- paper ex: P(a*ba*) --\n");
    // Alphabet {a=0, b=1}.  P(a*ba*) = {(m,1) | m in N}.
    // P(a*) = 0+<(1,0)>,  P(b) = {(0,1)},  P(a*) = 0+<(1,0)>
    // Mink: (0+<(1,0)>) + {(0,1)} = {(0,1)}+<(1,0)>
    // Mink with (0+<(1,0)>): {(0,1)}+<(1,0),(1,0)>  — both periods are (1,0)
    auto astar = re_star(re_letter(0));
    auto e     = re_concat(astar, re_concat(re_letter(1), astar));
    auto S     = parikh_image(e, 2);

    check( sl_contains(S, v({0,1})),  "P(a*ba*) contains (0,1)");
    check( sl_contains(S, v({1,1})),  "P(a*ba*) contains (1,1)");
    check( sl_contains(S, v({5,1})),  "P(a*ba*) contains (5,1)");
    check(!sl_contains(S, v({0,0})),  "P(a*ba*) does not contain (0,0)");
    check(!sl_contains(S, v({1,0})),  "P(a*ba*) does not contain (1,0)");
    check(!sl_contains(S, v({1,2})),  "P(a*ba*) does not contain (1,2)");
}

static void test_paper_ex2() {
    std::printf("-- paper ex: P((ab)*) -- (confirmed above, recheck) --\n");
    // Already covered, but also verify via the formula P(r)* where P(r) is finite:
    // P((ab)*) = <P(ab)> = <{(1,1)}> = {n*(1,1) | n in N}
    auto S = parikh_image(re_star(re_concat(re_letter(0), re_letter(1))), 2);
    check( sl_contains(S, v({3,3})), "P((ab)*) contains (3,3)");
    check(!sl_contains(S, v({3,4})), "P((ab)*) does not contain (3,4)");
}

// ── Suite 5: 3-letter alphabet ────────────────────────────────────────────────

static void test_three_letters() {
    std::printf("-- 3-letter: P(a*b*c*) --\n");
    // Alphabet {a=0, b=1, c=2}.
    // P(a*) = 0+<e0>,  P(b*) = 0+<e1>,  P(c*) = 0+<e2>
    // P(a*b*c*) = Mink of the above = 0 + <e0,e1,e2> = N^3
    auto e = re_concat(re_star(re_letter(0)),
                re_concat(re_star(re_letter(1)), re_star(re_letter(2))));
    auto S = parikh_image(e, 3);
    check( sl_contains(S, v({0,0,0})), "P(a*b*c*) contains (0,0,0)");
    check( sl_contains(S, v({2,0,0})), "P(a*b*c*) contains (2,0,0)");
    check( sl_contains(S, v({0,3,0})), "P(a*b*c*) contains (0,3,0)");
    check( sl_contains(S, v({1,2,3})), "P(a*b*c*) contains (1,2,3)");

    std::printf("-- 3-letter: P(ab+c) --\n");
    // P(ab) = {(1,1,0)},  P(c) = {(0,0,1)}
    // P(ab+c) = {(1,1,0), (0,0,1)}
    auto e2 = re_union(re_concat(re_letter(0), re_letter(1)), re_letter(2));
    auto S2 = parikh_image(e2, 3);
    check( sl_contains(S2, v({1,1,0})), "P(ab+c) contains (1,1,0)");
    check( sl_contains(S2, v({0,0,1})), "P(ab+c) contains (0,0,1)");
    check(!sl_contains(S2, v({0,0,0})), "P(ab+c) does not contain (0,0,0)");
    check(!sl_contains(S2, v({1,0,0})), "P(ab+c) does not contain (1,0,0)");
    check(!sl_contains(S2, v({1,1,1})), "P(ab+c) does not contain (1,1,1)");
}

// ── Suite 6: cross-check with NFA acceptance ─────────────────────────────────
// For a regex e, every word w accepted by the NFA should have Psi(w) in P(e).

static Vec parikh_of_word(const std::vector<int>& word, int k) {
    Vec v(k, 0);
    for (int a : word) v[a]++;
    return v;
}

static std::vector<int> w(const std::string& s) {
    std::vector<int> v;
    for (char c : s) v.push_back(c - 'a');
    return v;
}

static void test_cross_check() {
    std::printf("-- cross-check NFA vs Parikh image --\n");
    // e = a(b+c)* over {a=0,b=1,c=2}
    int k = 3;
    auto e   = re_concat(re_letter(0), re_star(re_union(re_letter(1), re_letter(2))));
    auto nfa = regex_to_nfa(e, k);
    auto S   = parikh_image(e, k);

    // Words accepted by NFA → their Parikh vectors must be in S
    std::vector<std::string> accept_words = {"a","ab","ac","abc","acb","abbc","abcbc"};
    for (auto& ws : accept_words) {
        auto word = w(ws);
        bool in_nfa = nfa.accepts(word);
        Vec pv = parikh_of_word(word, k);
        bool in_parikh = sl_contains(S, pv);
        check(in_nfa && in_parikh,
              ("NFA accepts '" + ws + "' and Parikh contains Psi('" + ws + "')").c_str());
    }

    // Parikh image may be larger (different words with same Parikh vector)
    // e.g. 'acb' and 'abc' have same Parikh image (1,1,1)
    Vec psi_abc = parikh_of_word(w("abc"), k);  // (1,1,1)
    Vec psi_acb = parikh_of_word(w("acb"), k);  // (1,1,1)
    check(psi_abc == psi_acb, "Psi(abc) == Psi(acb) -- same Parikh vector");
    check(sl_contains(S, psi_abc), "Parikh contains (1,1,1)");

    // Words rejected by NFA → their Parikh vectors need not be in S
    // but let's at least check some are not (e.g., pure-b words)
    Vec psi_b = parikh_of_word(w("b"), k);   // (0,1,0)
    check(!sl_contains(S, psi_b), "P(a(b+c)*) does not contain Psi('b')=(0,1,0)");
}

// ── Suite 7: idempotent and annihilator laws ──────────────────────────────────
static void test_laws() {
    std::printf("-- algebraic laws --\n");
    int k = 2;

    // P(0 + e) = P(e)
    auto e = re_letter(0);
    auto Se  = parikh_image(e, k);
    auto S0e = parikh_image(re_union(re_zero(), e), k);
    check( sl_contains(S0e, v({1,0})) ==  sl_contains(Se, v({1,0})), "P(0+a)=P(a) at (1,0)");
    check( sl_contains(S0e, v({0,0})) ==  sl_contains(Se, v({0,0})), "P(0+a)=P(a) at (0,0)");

    // P(0 * e) = P(0) = empty
    auto S0k = parikh_image(re_concat(re_zero(), e), k);
    check(S0k.empty(), "P(0*a)=empty");

    // P(eps * e) = P(e)
    auto Sek = parikh_image(re_concat(re_one(), e), k);
    check( sl_contains(Sek, v({1,0})), "P(eps*a) contains (1,0)");
    check(!sl_contains(Sek, v({0,0})), "P(eps*a) does not contain (0,0)");

    // P((e*)* ) = P(e*) -- star idempotent on Parikh level
    auto Sdstar  = parikh_image(re_star(re_star(re_letter(0))), k);
    auto Sstar   = parikh_image(re_star(re_letter(0)), k);
    // Both should contain exactly {n*(1,0) | n in N}
    check( sl_contains(Sdstar, v({0,0})) &&  sl_contains(Sstar, v({0,0})), "(a*)*: (0,0)");
    check( sl_contains(Sdstar, v({3,0})) &&  sl_contains(Sstar, v({3,0})), "(a*)*: (3,0)");
    check(!sl_contains(Sdstar, v({0,1})) && !sl_contains(Sstar, v({0,1})), "(a*)*: not (0,1)");
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    std::printf("=== base cases ===\n");
    test_zero();
    test_epsilon();
    test_letter();

    std::printf("\n=== union and concat ===\n");
    test_union();
    test_concat();

    std::printf("\n=== star ===\n");
    test_star_letter();
    test_star_empty();
    test_star_union();
    test_star_concat();

    std::printf("\n=== paper examples ===\n");
    test_paper_ex1();
    test_paper_ex2();

    std::printf("\n=== 3-letter alphabet ===\n");
    test_three_letters();

    std::printf("\n=== cross-check with NFA ===\n");
    test_cross_check();

    std::printf("\n=== algebraic laws ===\n");
    test_laws();

    std::printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}

#pragma once
#include "nfa.h"
#include <memory>
#include <set>
#include <string>

// Regular expression AST node
struct Regex {
    enum class Kind { Zero, One, Letter, Union, Concat, Star };
    Kind kind;
    int letter = 0;                       // for Letter
    std::shared_ptr<Regex> left, right;   // for Union, Concat
    std::shared_ptr<Regex> child;         // for Star
    std::string id;                       // canonical string — used as state key
};

using RegexPtr = std::shared_ptr<Regex>;

// Strict-weak ordering on RegexPtr by canonical id, for use in std::set/map
struct RegexCmp {
    bool operator()(const RegexPtr& a, const RegexPtr& b) const {
        return a->id < b->id;
    }
};
using RegexSet = std::set<RegexPtr, RegexCmp>;

// ── Factory functions (apply simplification rules) ───────────────────────────
//
//   re_zero()              ∅
//   re_one()               ε
//   re_letter(a)           a
//   re_union(r, s)         r + s    (0+e=e, e+0=e, e+e=e; commutative canonical order)
//   re_concat(r, s)        r s      (0·e=0, e·0=0, ε·e=e, e·ε=e)
//   re_star(r)             r*       (0*=ε, ε*=ε, (r*)*=r*)
//   re_plus(r)             r+  = r·r*

RegexPtr re_zero();
RegexPtr re_one();
RegexPtr re_letter(int a);
RegexPtr re_union(RegexPtr a, RegexPtr b);
RegexPtr re_concat(RegexPtr a, RegexPtr b);
RegexPtr re_star(RegexPtr e);
RegexPtr re_plus(RegexPtr e);

// ── Semantic helpers ─────────────────────────────────────────────────────────

// ν(r): true iff ε ∈ L(r)
bool nullable(const RegexPtr& e);

// Antimirov partial derivative ∂_a(r):
//   returns the set of expressions {t} such that L(r) ∩ aΣ* = a · ∪_t L(t)
//
// Rules:
//   ∂_a(∅)   = ∅
//   ∂_a(ε)   = ∅
//   ∂_a(b)   = {ε}  if a=b,  ∅ otherwise
//   ∂_a(r+s) = ∂_a(r) ∪ ∂_a(s)
//   ∂_a(rs)  = {t·s | t ∈ ∂_a(r)} ∪ (ν(r) ? ∂_a(s) : ∅)
//   ∂_a(r*)  = {t·r* | t ∈ ∂_a(r)}
RegexSet partial_deriv(const RegexPtr& e, int a);

// ── NFA construction ─────────────────────────────────────────────────────────
//
// Build the Antimirov NFA for regex `e` over alphabet {0,...,num_letters-1}:
//   Q     = all expressions reachable by iterated partial derivatives from e
//   q0    = e
//   δ(q,a)= ∂_a(q)   (non-deterministic: one transition per element of the set)
//   F     = { q ∈ Q | ν(q) }
NFA regex_to_nfa(const RegexPtr& e, int num_letters);

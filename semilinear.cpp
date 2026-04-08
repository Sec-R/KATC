#include "semilinear.h"
#include "nfa.h"
#include <algorithm>
#include <cassert>

// ── Vector helpers ────────────────────────────────────────────────────────────

static Vec zero_vec(int k) { return Vec(k, 0); }

static Vec unit_vec(int k, int i) {
    Vec v(k, 0);
    v[i] = 1;
    return v;
}

static Vec vec_add(const Vec& a, const Vec& b) {
    int k = a.size();
    Vec r(k);
    for (int i = 0; i < k; i++) r[i] = a[i] + b[i];
    return r;
}

static bool vec_is_zero(const Vec& v) {
    for (auto x : v) if (x != 0) return false;
    return true;
}

// ── Semilinear set operations ─────────────────────────────────────────────────

// Remove all-zero period vectors (redundant generators).
static void prune_zero_periods(LinearSet& L) {
    L.periods.erase(
        std::remove_if(L.periods.begin(), L.periods.end(), vec_is_zero),
        L.periods.end());
}

// Minkowski sum of two semilinear sets:
//   (b + <P>) + (c + <Q>) = (b+c) + <P,Q>
// The result is the cross-product of all pairs of linear sets.
static SemilinearSet sl_mink_sum(const SemilinearSet& S1, const SemilinearSet& S2) {
    SemilinearSet result;
    for (auto& L1 : S1) {
        for (auto& L2 : S2) {
            LinearSet L;
            L.base = vec_add(L1.base, L2.base);
            L.periods = L1.periods;
            L.periods.insert(L.periods.end(), L2.periods.begin(), L2.periods.end());
            result.push_back(std::move(L));
        }
    }
    return result;
}

// Star of a semilinear set:
//   (union_l L_l)* = 0 + < b_1,...,b_t, p_{1,1},...,p_{t,m_t} >
// where b_l are the bases and p_{l,j} the period vectors of each L_l.
// Proof sketch: any finite sum of elements from S is
//   sum_l c_l * b_l + sum_{l,j} k_{l,j} p_{l,j}  with c_l, k_{l,j} in N,
// which ranges exactly over the linear set above.
// Special case: (empty)* = { 0^k } = the epsilon word.
static SemilinearSet sl_star(const SemilinearSet& S, int k) {
    LinearSet L;
    L.base = zero_vec(k);
    for (auto& Ls : S) {
        L.periods.push_back(Ls.base);
        for (auto& p : Ls.periods)
            L.periods.push_back(p);
    }
    prune_zero_periods(L);
    return {L};
}

// ── Parikh image ──────────────────────────────────────────────────────────────

SemilinearSet parikh_image(const RegexPtr& e, int k) {
    switch (e->kind) {

        case Regex::Kind::Zero:
            return {};                                            // P(0) = empty

        case Regex::Kind::One:
            return { LinearSet{zero_vec(k), {}} };               // P(eps) = {0^k}

        case Regex::Kind::Letter:
            return { LinearSet{unit_vec(k, e->letter), {}} };    // P(a_i) = {e_i}

        case Regex::Kind::Union: {
            // P(r+s) = P(r) union P(s)
            auto S1 = parikh_image(e->left,  k);
            auto S2 = parikh_image(e->right, k);
            S1.insert(S1.end(), S2.begin(), S2.end());
            return S1;
        }

        case Regex::Kind::Concat:
            // P(rs) = P(r) Minkowski-sum P(s)
            return sl_mink_sum(parikh_image(e->left, k), parikh_image(e->right, k));

        case Regex::Kind::Star:
            // P(r*) = star(P(r))
            return sl_star(parikh_image(e->child, k), k);
    }
    return {};  // unreachable
}

// ── Membership test (for testing only) ───────────────────────────────────────
//
// Check if `remainder` is a non-negative integer combination of periods[idx..end).
// Tries all valid coefficients n in [0, floor(remainder[i] / period[i])] greedily.
static bool can_represent(const std::vector<Vec>& periods, int idx, Vec remainder) {
    // Prune: negative component → impossible
    for (auto x : remainder) if (x < 0) return false;

    if (idx == (int)periods.size()) {
        return vec_is_zero(remainder);
    }

    const Vec& p = periods[idx];

    if (vec_is_zero(p)) {
        // Zero period contributes nothing; skip it.
        return can_represent(periods, idx + 1, remainder);
    }

    // Compute the tightest upper bound on n:
    // need remainder[i] - n*p[i] >= 0 for all i, so n <= floor(remainder[i]/p[i]) when p[i]>0.
    int64_t upper = INT64_MAX;
    for (int i = 0; i < (int)p.size(); i++)
        if (p[i] > 0)
            upper = std::min(upper, remainder[i] / p[i]);
    if (upper == INT64_MAX) upper = 0;  // p has no positive components

    for (int64_t n = 0; n <= upper; n++) {
        Vec r = remainder;
        for (int i = 0; i < (int)r.size(); i++) r[i] -= n * p[i];
        if (can_represent(periods, idx + 1, r)) return true;
    }
    return false;
}

static bool linear_set_contains(const LinearSet& L, const Vec& v) {
    assert(v.size() == L.base.size());
    // Compute r = v - b; if any component goes negative, v is not in L.
    Vec r = v;
    for (int i = 0; i < (int)r.size(); i++) {
        r[i] -= L.base[i];
        if (r[i] < 0) return false;
    }
    return can_represent(L.periods, 0, r);
}

bool sl_contains(const SemilinearSet& S, const Vec& v) {
    for (auto& L : S)
        if (linear_set_contains(L, v)) return true;
    return false;
}

// ── Parikh image of NFA ───────────────────────────────────────────────────────
//
// Kleene's path algorithm on semilinear matrices.
// R[i][j] = semilinear set of Parikh vectors of all paths from state i to j.
//
// Step k: R[i][j] += R[i][k] ⊕ star(R[k][k]) ⊕ R[k][j]
//
// This is correct because P(L1·L2) = P(L1) ⊕ P(L2) and P(L*) = star(P(L)).

SemilinearSet parikh_image_nfa(const NFA& nfa, int k) {
    int n = nfa.num_states;
    if (n == 0) return {};

    using SLMatrix = std::vector<std::vector<SemilinearSet>>;
    SLMatrix R(n, std::vector<SemilinearSet>(n));

    // Identity: empty path from each state to itself has Parikh vector 0.
    Vec zero = zero_vec(k);
    for (int i = 0; i < n; i++)
        R[i][i].push_back(LinearSet{zero, {}});

    // Direct transitions.
    for (int i = 0; i < n; i++) {
        for (auto& [letter, targets] : nfa.trans[i]) {
            Vec v = (letter == EPSILON) ? zero_vec(k) : unit_vec(k, letter);
            LinearSet ls{v, {}};
            for (int j : targets)
                R[i][j].push_back(ls);
        }
    }

    // Kleene's path algorithm: admit state `wp` as waypoint.
    for (int wp = 0; wp < n; wp++) {
        // Snapshot R[wp][wp] before we overwrite R this iteration.
        SemilinearSet Rkk_star = sl_star(R[wp][wp], k);

        SLMatrix R2(n, std::vector<SemilinearSet>(n));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                R2[i][j] = R[i][j];  // paths not through wp
                if (!R[i][wp].empty() && !R[wp][j].empty()) {
                    // R[i][wp] ⊕ Rkk* ⊕ R[wp][j]
                    SemilinearSet via = sl_mink_sum(
                        sl_mink_sum(R[i][wp], Rkk_star), R[wp][j]);
                    for (auto& ls : via)
                        R2[i][j].push_back(ls);
                }
            }
        }
        R = std::move(R2);
    }

    // Union over all (initial, final) pairs.
    SemilinearSet result;
    for (int q0 : nfa.initial)
        for (int qf : nfa.final_states)
            for (const LinearSet& ls : R[q0][qf])
                result.push_back(ls);
    return result;
}

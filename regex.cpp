#include "regex.h"
#include <map>
#include <queue>

// ── Factory functions ────────────────────────────────────────────────────────

RegexPtr re_zero() {
    auto r  = std::make_shared<Regex>();
    r->kind = Regex::Kind::Zero;
    r->id   = "0";
    return r;
}

RegexPtr re_one() {
    auto r  = std::make_shared<Regex>();
    r->kind = Regex::Kind::One;
    r->id   = "1";
    return r;
}

RegexPtr re_letter(int a) {
    auto r    = std::make_shared<Regex>();
    r->kind   = Regex::Kind::Letter;
    r->letter = a;
    r->id     = "l" + std::to_string(a);
    return r;
}

RegexPtr re_union(RegexPtr a, RegexPtr b) {
    // 0 + e = e,  e + 0 = e
    if (a->kind == Regex::Kind::Zero) return b;
    if (b->kind == Regex::Kind::Zero) return a;
    // e + e = e
    if (a->id == b->id) return a;
    // Canonical order (union is commutative): sort by id so r+s = s+r
    if (a->id > b->id) std::swap(a, b);
    auto r  = std::make_shared<Regex>();
    r->kind = Regex::Kind::Union;
    r->left = a; r->right = b;
    r->id   = "(+" + a->id + "," + b->id + ")";
    return r;
}

RegexPtr re_concat(RegexPtr a, RegexPtr b) {
    // 0·e = 0,  e·0 = 0
    if (a->kind == Regex::Kind::Zero || b->kind == Regex::Kind::Zero) return re_zero();
    // ε·e = e,  e·ε = e
    if (a->kind == Regex::Kind::One) return b;
    if (b->kind == Regex::Kind::One) return a;
    auto r  = std::make_shared<Regex>();
    r->kind = Regex::Kind::Concat;
    r->left = a; r->right = b;
    r->id   = "(." + a->id + "," + b->id + ")";
    return r;
}

RegexPtr re_star(RegexPtr e) {
    // 0* = ε,  ε* = ε
    if (e->kind == Regex::Kind::Zero || e->kind == Regex::Kind::One) return re_one();
    // (r*)* = r*
    if (e->kind == Regex::Kind::Star) return e;
    auto r  = std::make_shared<Regex>();
    r->kind = Regex::Kind::Star;
    r->child = e;
    r->id    = "(*" + e->id + ")";
    return r;
}

RegexPtr re_plus(RegexPtr e) {
    return re_concat(e, re_star(e));
}

// ── Semantic helpers ─────────────────────────────────────────────────────────

bool nullable(const RegexPtr& e) {
    switch (e->kind) {
        case Regex::Kind::Zero:   return false;
        case Regex::Kind::One:    return true;
        case Regex::Kind::Letter: return false;
        case Regex::Kind::Union:  return nullable(e->left) || nullable(e->right);
        case Regex::Kind::Concat: return nullable(e->left) && nullable(e->right);
        case Regex::Kind::Star:   return true;
    }
    return false; // unreachable
}

RegexSet partial_deriv(const RegexPtr& e, int a) {
    RegexSet result;
    switch (e->kind) {
        case Regex::Kind::Zero:   // ∂_a(∅) = ∅
        case Regex::Kind::One:    // ∂_a(ε) = ∅
            break;

        case Regex::Kind::Letter:
            // ∂_a(b) = {ε} if a=b, else ∅
            if (e->letter == a)
                result.insert(re_one());
            break;

        case Regex::Kind::Union:
            // ∂_a(r+s) = ∂_a(r) ∪ ∂_a(s)
            for (auto& t : partial_deriv(e->left,  a)) result.insert(t);
            for (auto& t : partial_deriv(e->right, a)) result.insert(t);
            break;

        case Regex::Kind::Concat:
            // ∂_a(rs) = {t·s | t ∈ ∂_a(r)} ∪ (ν(r) ? ∂_a(s) : ∅)
            for (auto& t : partial_deriv(e->left, a))
                result.insert(re_concat(t, e->right));
            if (nullable(e->left))
                for (auto& t : partial_deriv(e->right, a))
                    result.insert(t);
            break;

        case Regex::Kind::Star:
            // ∂_a(r*) = {t·r* | t ∈ ∂_a(r)}
            for (auto& t : partial_deriv(e->child, a))
                result.insert(re_concat(t, e));
            break;
    }
    return result;
}

// ── NFA construction ─────────────────────────────────────────────────────────

NFA regex_to_nfa(const RegexPtr& e, int num_letters) {
    // Map canonical id → state index; worklist for BFS over reachable expressions
    std::map<std::string, int> state_map;
    std::vector<RegexPtr>      states;
    std::queue<RegexPtr>       worklist;

    // Register an expression as a state (no-op if already seen)
    auto get_or_add = [&](const RegexPtr& r) -> int {
        auto [it, inserted] = state_map.emplace(r->id, (int)states.size());
        if (inserted) {
            states.push_back(r);
            worklist.push(r);
        }
        return it->second;
    };

    // Initial state is the regex itself
    int init = get_or_add(e);

    // BFS: for each reachable expression, compute all partial derivatives
    struct RawTrans { int from, letter, to; };
    std::vector<RawTrans> raw;

    while (!worklist.empty()) {
        RegexPtr cur     = worklist.front(); worklist.pop();
        int      cur_idx = state_map.at(cur->id);

        for (int a = 0; a < num_letters; a++)
            for (auto& d : partial_deriv(cur, a))
                raw.push_back({cur_idx, a, get_or_add(d)});
    }

    // Assemble NFA
    int n = (int)states.size();
    NFA nfa(n);
    nfa.initial = {init};

    for (int i = 0; i < n; i++)
        if (nullable(states[i]))
            nfa.final_states.insert(i);

    for (auto& [from, letter, to] : raw)
        nfa.add_transition(from, letter, to);

    return nfa;
}

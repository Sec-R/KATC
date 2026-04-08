#include "nfa.h"
#include <queue>

void NFA::add_transition(int from, int letter, int to) {
    trans[from][letter].insert(to);
}

std::set<int> NFA::epsilon_closure(const std::set<int>& states) const {
    std::set<int> closure = states;
    std::queue<int> wl;
    for (int s : states) wl.push(s);
    while (!wl.empty()) {
        int s = wl.front(); wl.pop();
        auto it = trans[s].find(EPSILON);
        if (it != trans[s].end()) {
            for (int t : it->second)
                if (closure.insert(t).second)
                    wl.push(t);
        }
    }
    return closure;
}

bool NFA::accepts(const std::vector<int>& word) const {
    std::set<int> cur = epsilon_closure(initial);
    for (int a : word) {
        std::set<int> nxt;
        for (int s : cur) {
            auto it = trans[s].find(a);
            if (it != trans[s].end())
                nxt.insert(it->second.begin(), it->second.end());
        }
        cur = epsilon_closure(nxt);
    }
    for (int s : cur)
        if (final_states.count(s)) return true;
    return false;
}

bool NFA::is_empty() const {
    // BFS over all transitions (including ε) to find any reachable final state
    std::set<int> visited;
    std::queue<int> wl;
    for (int s : initial)
        if (visited.insert(s).second) wl.push(s);
    while (!wl.empty()) {
        int s = wl.front(); wl.pop();
        if (final_states.count(s)) return false;
        for (auto& [letter, nexts] : trans[s])
            for (int t : nexts)
                if (visited.insert(t).second) {
                    if (final_states.count(t)) return false;
                    wl.push(t);
                }
    }
    return true;
}

bool NFA::accepts_epsilon() const {
    for (int s : epsilon_closure(initial))
        if (final_states.count(s)) return true;
    return false;
}

// ---- Factory functions ----

NFA make_empty_nfa() {
    NFA n(1);
    n.initial = {0};
    // no final states → accepts ∅
    return n;
}

NFA make_epsilon_nfa() {
    NFA n(1);
    n.initial    = {0};
    n.final_states = {0};
    return n;
}

NFA make_letter_nfa(int a) {
    NFA n(2);
    n.initial    = {0};
    n.final_states = {1};
    n.add_transition(0, a, 1);
    return n;
}

// ---- Helper: copy NFA b into `result`, offsetting all state indices by `offset` ----
static void copy_nfa_into(NFA& result, const NFA& b, int offset) {
    for (int i = 0; i < b.num_states; i++)
        for (auto& [letter, nexts] : b.trans[i])
            for (int t : nexts)
                result.add_transition(i + offset, letter, t + offset);
}

// ---- NFA operations ----

NFA nfa_union(const NFA& a, const NFA& b) {
    NFA result(a.num_states + b.num_states);
    copy_nfa_into(result, a, 0);
    copy_nfa_into(result, b, a.num_states);
    for (int s : a.initial)      result.initial.insert(s);
    for (int s : b.initial)      result.initial.insert(s + a.num_states);
    for (int s : a.final_states) result.final_states.insert(s);
    for (int s : b.final_states) result.final_states.insert(s + a.num_states);
    return result;
}

NFA nfa_concat(const NFA& a, const NFA& b) {
    NFA result(a.num_states + b.num_states);
    copy_nfa_into(result, a, 0);
    copy_nfa_into(result, b, a.num_states);
    result.initial = a.initial;
    // ε-transitions from a's final states to b's initial states
    for (int af : a.final_states)
        for (int bi : b.initial)
            result.add_transition(af, EPSILON, bi + a.num_states);
    for (int s : b.final_states) result.final_states.insert(s + a.num_states);
    return result;
}

NFA nfa_star(const NFA& a) {
    // New state 0 is both initial and final (accepts ε).
    // All of a's states are offset by 1.
    NFA result(a.num_states + 1);
    copy_nfa_into(result, a, 1);
    result.initial     = {0};
    result.final_states = {0};
    // ε: new state → a's initial states
    for (int s : a.initial)      result.add_transition(0, EPSILON, s + 1);
    // ε: a's final states → new state
    for (int s : a.final_states) result.add_transition(s + 1, EPSILON, 0);
    return result;
}

NFA nfa_plus(const NFA& a) {
    return nfa_concat(a, nfa_star(a));
}


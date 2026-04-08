#pragma once
#include <functional>
#include <map>
#include <set>
#include <vector>

// Epsilon transition is represented by letter value -1
static constexpr int EPSILON = -1;

// NFA over an integer alphabet {0, 1, ..., k-1}
struct NFA {
    int num_states = 0;
    std::set<int> initial;
    std::set<int> final_states;
    // trans[state][letter] = set of successor states; letter == EPSILON for ε
    std::vector<std::map<int, std::set<int>>> trans;

    NFA() = default;
    explicit NFA(int n) : num_states(n), trans(n) {}

    void add_transition(int from, int letter, int to);

    // Compute ε-closure of a set of states
    std::set<int> epsilon_closure(const std::set<int>& states) const;

    // Simulate the NFA on `word`; return true if accepted
    bool accepts(const std::vector<int>& word) const;

    // Return true if L(NFA) = ∅
    bool is_empty() const;

    // Return true if ε ∈ L(NFA)
    bool accepts_epsilon() const;
};

// --- NFA factory functions ---
NFA make_empty_nfa();       // Accepts ∅
NFA make_epsilon_nfa();     // Accepts {ε}
NFA make_letter_nfa(int a); // Accepts {a}

// --- NFA operations (produce new NFA with renumbered states) ---
NFA nfa_union(const NFA& a, const NFA& b);   // L(a) ∪ L(b)
NFA nfa_concat(const NFA& a, const NFA& b);  // L(a) · L(b)
NFA nfa_star(const NFA& a);                   // L(a)*
NFA nfa_plus(const NFA& a);                   // L(a)+ = L(a) · L(a)*

// ── Semiring for generic matrix operations ────────────────────────────────────
//
// A semiring (T, +, ·, 0, 1) together with a star operation.
// Used to parameterize matrix_star and friends.

template<typename T>
struct MatSR {
    T                        zero;
    std::function<T(T,T)>    add;
    std::function<T(T,T)>    mul;
    std::function<T(T)>      star;
};

// ── Matrix type ───────────────────────────────────────────────────────────────
//
// NfaMatrix<T> is an n×m matrix of semiring elements.
// For the factorization step: T = RegexPtr.
// For the symbolic step:      T = SemilinearSet (built separately).

template<typename T>
using NfaMatrix = std::vector<std::vector<T>>;

// ── Generic matrix operations ─────────────────────────────────────────────────

// All-zero matrix of given dimensions.
template<typename T>
NfaMatrix<T> mat_zero(int rows, int cols, const T& zero_val) {
    return NfaMatrix<T>(rows, std::vector<T>(cols, zero_val));
}

// C[i][j] = sr.add(A[i][j], B[i][j])
template<typename T>
NfaMatrix<T> mat_add(const NfaMatrix<T>& A, const NfaMatrix<T>& B,
                      const MatSR<T>& sr) {
    int r = (int)A.size(), c = (int)A[0].size();
    NfaMatrix<T> C = A;
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            C[i][j] = sr.add(C[i][j], B[i][j]);
    return C;
}

// C[i][j] = sum_k sr.mul(A[i][k], B[k][j])   (A: r×p, B: p×c)
template<typename T>
NfaMatrix<T> mat_mul(const NfaMatrix<T>& A, const NfaMatrix<T>& B,
                      const MatSR<T>& sr) {
    int r = (int)A.size(), p = (int)A[0].size(), c = (int)B[0].size();
    NfaMatrix<T> C = mat_zero<T>(r, c, sr.zero);
    for (int i = 0; i < r; i++)
        for (int k = 0; k < p; k++)
            for (int j = 0; j < c; j++)
                C[i][j] = sr.add(C[i][j], sr.mul(A[i][k], B[k][j]));
    return C;
}

// Submatrix rows [r0,r1) × cols [c0,c1)
template<typename T>
NfaMatrix<T> submat(const NfaMatrix<T>& M, int r0, int r1, int c0, int c1) {
    NfaMatrix<T> S(r1 - r0, std::vector<T>(c1 - c0, M[r0][c0]));
    for (int i = r0; i < r1; i++)
        for (int j = c0; j < c1; j++)
            S[i - r0][j - c0] = M[i][j];
    return S;
}

// Assemble [[A, B], [C, D]] into one matrix.
template<typename T>
NfaMatrix<T> mat_assemble(const NfaMatrix<T>& A, const NfaMatrix<T>& B,
                            const NfaMatrix<T>& C, const NfaMatrix<T>& D) {
    int ra = (int)A.size(), ca = (int)A[0].size();
    int rb = (int)C.size(), cb = (int)B[0].size();
    NfaMatrix<T> M(ra + rb, std::vector<T>(ca + cb, A[0][0]));
    for (int i = 0; i < ra; i++)
        for (int j = 0; j < ca; j++) M[i][j]           = A[i][j];
    for (int i = 0; i < ra; i++)
        for (int j = 0; j < cb; j++) M[i][ca + j]       = B[i][j];
    for (int i = 0; i < rb; i++)
        for (int j = 0; j < ca; j++) M[ra + i][j]       = C[i][j];
    for (int i = 0; i < rb; i++)
        for (int j = 0; j < cb; j++) M[ra + i][ca + j]  = D[i][j];
    return M;
}

// ── matrix_star: 2×2 block formula (recursive) ────────────────────────────────
//
// Split M into blocks:  M = [[A, B], [C, D]]   (top p rows, bottom q rows)
//
// M* = [[E,      F     ],
//        [G,      H     ]]
//
// where:
//   Ds = D*
//   E  = (A + B·Ds·C)*
//   F  = E·B·Ds
//   G  = Ds·C·E
//   H  = Ds + G·B·Ds
//
// Base case (1×1): M* = [[star(M[0][0])]]

template<typename T>
NfaMatrix<T> matrix_star(const NfaMatrix<T>& M, const MatSR<T>& sr) {
    int n = (int)M.size();
    if (n == 1)
        return {{sr.star(M[0][0])}};

    int p = n / 2;   // top-block height/width
    int q = n - p;   // bottom-block height/width

    NfaMatrix<T> A = submat(M, 0, p, 0, p);   // p×p
    NfaMatrix<T> B = submat(M, 0, p, p, n);   // p×q
    NfaMatrix<T> C = submat(M, p, n, 0, p);   // q×p
    NfaMatrix<T> D = submat(M, p, n, p, n);   // q×q

    NfaMatrix<T> Ds    = matrix_star(D, sr);               // q×q
    NfaMatrix<T> BDs   = mat_mul(B, Ds, sr);               // p×q
    NfaMatrix<T> DsC   = mat_mul(Ds, C, sr);               // q×p
    NfaMatrix<T> E     = matrix_star(mat_add(A, mat_mul(BDs, C, sr), sr), sr); // p×p
    NfaMatrix<T> F     = mat_mul(E, BDs, sr);              // p×q
    NfaMatrix<T> G     = mat_mul(DsC, E, sr);              // q×p
    NfaMatrix<T> H     = mat_add(Ds, mat_mul(G, BDs, sr), sr); // q×q

    return mat_assemble(E, F, G, H);
}

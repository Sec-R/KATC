# KA+C Decision Procedure

Implementation of a decision procedure for **Kleene Algebra with Transitive
Commutativity Conditions (KA+C)** in C++17.  Given a commutativity partition
of the alphabet and two regular expressions, the procedure checks whether they
are equal under all commutativity axioms implied by the partition.

---

## Build Environment Setup

### Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Visual Studio 2022 | Community or above | MSVC compiler (`cl.exe`) |
| CMake | ≥ 3.14 | bundled with VS, or from cmake.org |
| Ninja | any | bundled with VS |
| LLVM/MLIR | built from source | see below |

### 1. Build LLVM/MLIR from source

Clone LLVM and build with the Presburger analysis library enabled:

```bat
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

cmake -S llvm -B build ^
  -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DLLVM_ENABLE_PROJECTS="mlir" ^
  -DLLVM_TARGETS_TO_BUILD="X86" ^
  -DLLVM_BUILD_TESTS=OFF ^
  -DMLIR_BUILD_MLIR_C_DYLIB=OFF

cmake --build build --target MLIRPresburger
```

This produces `build/lib/cmake/mlir/` which CMake uses to locate MLIR.

### 2. Configure the KATC project

Open a **Visual Studio 2022 x64 Native Tools Command Prompt** (this sets
`INCLUDE`, `LIB`, `PATH` for `cl.exe` and `link.exe`), then:

```bat
cd path\to\KATC\cpp
cmake -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DMLIR_DIR=path\to\llvm-project\build\lib\cmake\mlir
```

### 3. Build all targets

```bat
cmake --build build
```

This produces four test executables in `build\`:

| Executable | Tests |
|---|---|
| `test_factorization.exe` | NFA ops, matrix star, factorization automaton |
| `test_regex.exe` | Antimirov derivatives, regex-to-NFA |
| `test_parikh.exe` | Parikh image (semilinear), Presburger conversion |
| `test_symbolic_equiv.exe` | Minterms, symbolic equivalence, full pipeline |

### 4. Run tests

```bat
cd build
ctest --output-on-failure
```

Each test has a **30-second timeout**.  If a test hangs, something has gone
wrong (likely an expensive Presburger complement computation).

### Convenience script

The file `build\build_all.bat` re-runs the VS environment setup and calls
`cmake --build .` in one step:

```bat
build\build_all.bat
```

---

## Architecture Overview

The decision procedure runs in three phases, each producing an intermediate
automaton whose language captures the C-equivalence classes of words:

```
RegexPtr ──► NFA<int> ──► FactorizationAutomaton ──► SymbolicAutomaton
                                                           │
                                               check_equiv(A, B) → bool
```

### Phase 1 – Regular NFA  (`nfa.h / nfa.cpp`, `regex.h / regex.cpp`)

A standard NFA over integer letters `{0, …, k-1}`.

**`regex_to_nfa(e, k)`** — Antimirov partial-derivative construction.
- States = reachable sub-expressions of `e` (BFS over `partial_deriv`).
- Transition `q –a→ r` for each `r ∈ ∂_a(q)`.
- Final states = nullable expressions.
- Produces a small NFA without ε-transitions.

**`partial_deriv(e, a)`** — Antimirov derivative ∂_a(e):
```
∂_a(∅)   = ∅
∂_a(ε)   = ∅
∂_a(b)   = {ε} if a=b, else ∅
∂_a(r+s) = ∂_a(r) ∪ ∂_a(s)
∂_a(rs)  = {t·s | t ∈ ∂_a(r)} ∪ (ν(r) ? ∂_a(s) : ∅)
∂_a(r*)  = {t·r* | t ∈ ∂_a(r)}
```

**NFA operations** (`make_empty_nfa`, `make_epsilon_nfa`, `make_letter_nfa`,
`nfa_union`, `nfa_concat`, `nfa_star`, `nfa_plus`) — Thompson constructions
with ε-transitions and state renumbering.

---

### Phase 2 – Factorization Automaton  (`factorization.h / factorization.cpp`)

Implements the **C-factorization automaton** from the KA+C paper.

**Alphabet partition** — `Partition` assigns each letter to an equivalence
class `0, …, m-1`.  Letters in the same class commute with each other.

**State encoding** — states are pairs `(q, k)` where `q` is an original NFA
state and `k ∈ {0,…,m-1}` is the "current class".  Encoded as `q*m + k`.

**`build_factorization_automaton(nfa, part)`**:

For each destination class `k2`:
1. Build the `n×n` **regex matrix** `A_{k2}` where entry `[i][j]` is the
   union of `re_letter(a)` for all letters `a ∈ Σ_{k2}` with transition
   `(i, a, j)` in the NFA.
2. Compute `A_{k2}^+ = A_{k2} · (A_{k2}*)` using `matrix_star` (block
   formula, see below).  This gives the regex for all non-empty paths using
   only class-`k2` letters.
3. For each `(q1, q2)` with `A_{k2}^+[q1][q2] ≠ ∅`, and each source class
   `k1 ≠ k2`, emit transition `(encode(q1,k1), label, encode(q2,k2))` where
   `label = A_{k2}^+[q1][q2]` as a `RegexPtr`.

**Initial states** = `nfa.initial × {0,…,m-1}`.  
**Final states** = `nfa.final_states × {0,…,m-1}`.

---

### Generic Matrix Semiring  (`nfa.h`)

All matrix computations are parameterised by a **semiring** type `T`:

```cpp
template<typename T>
struct MatSR {
    T                     zero;
    std::function<T(T,T)> add;   // semiring +
    std::function<T(T,T)> mul;   // semiring ·
    std::function<T(T)>   star;  // Kleene star
};
```

`NfaMatrix<T>` = `std::vector<std::vector<T>>` (row-major `n×n` matrix).

**`matrix_star(M, sr)`** — Kleene star of an `n×n` matrix using the
recursive **2×2 block formula**.  For `M = [[A,B],[C,D]]` (splitting rows/
columns at midpoint `p = n/2`):

```
Ds = D*              (recursive call, (n-p)×(n-p))
E  = (A + B·Ds·C)*   (recursive call, p×p)
F  = E·B·Ds
G  = Ds·C·E
H  = Ds + G·B·Ds

M* = [[E, F],
      [G, H]]
```

Base case: `n=1` → `[[sr.star(M[0][0])]]`.

**For the factorization step**, the semiring is `MatSR<RegexPtr>`:
- `zero` = `re_zero()`
- `add(a,b)` = `re_union(a,b)`
- `mul(a,b)` = `re_concat(a,b)`
- `star(a)` = `re_star(a)`

---

### Phase 3 – Symbolic Automaton  (`symbolic_equiv.h / symbolic_equiv.cpp`)

Replaces each `RegexPtr` label on the factorization automaton with its
**Parikh image** as a `PresburgerSet` (a Presburger-arithmetic formula over
`Z^k`).

**`fac_to_symbolic(fac, k)`** — For each factorization transition `(from,
label, to)`:
1. Call `parikh_image(label, k)` → `SemilinearSet`.
2. Convert to `PresburgerSet` via `to_presburger_set`.
3. Skip if the resulting set is integer-empty.

**`parikh_image(e, k)`** — Computes the Parikh image of a regex structurally:
```
P(∅)   = ∅
P(ε)   = {0^k}
P(a_i) = {e_i}               (unit vector)
P(r+s) = P(r) ∪ P(s)
P(rs)  = P(r) ⊕ P(s)         (Minkowski sum)
P(r*)  = sl_star(P(r))        = {0^k} + <all bases and periods of P(r)>
```

---

### Semilinear Sets  (`semilinear.h / semilinear.cpp`)

```cpp
struct LinearSet   { Vec base; std::vector<Vec> periods; };
using  SemilinearSet = std::vector<LinearSet>;
```

A **linear set** `b + <p_1,…,p_m>` = `{ b + Σ nᵢpᵢ | nᵢ ∈ ℕ }`.

Key operations:
- **`sl_mink_sum(S1, S2)`** — Minkowski sum: cross-product of all linear-set
  pairs, concatenating base and periods.
- **`sl_star(S)`** — Semilinear star: `{0} + <all bases and periods of S>`.
- **`sl_contains(S, v)`** — Membership test (bounded backtracking, for tests).

---

### Presburger Conversion  (`parikh_mlir.h / parikh_mlir.cpp`)

**`to_presburger_set(S, k)`** — Converts a `SemilinearSet` to an MLIR
`PresburgerSet`.

Each linear set `b + <p_1,…,p_m>` becomes one `IntegerPolyhedron` with:
- `k` **dimension** variables `x_0,…,x_{k-1}` (Parikh vector components).
- `m` **local** (existentially quantified) variables `n_0,…,n_{m-1}`.

Constraints:
```
x_i = b_i + Σ_j nⱼ·pⱼ[i]    for each i      (equality)
nⱼ ≥ 0                        for each j      (inequality)
```

The `PresburgerSet` is the union of all such polyhedra, with locals projected
out automatically by MLIR's Presburger library.

---

### Equivalence Check  (`symbolic_equiv.h / symbolic_equiv.cpp`)

**`compute_minterms(labels, dim)`** — Partition-refinement over the label
sets.  No "universe" atom is created; only points that appear in at least one
label are tracked (points outside all labels map both automata to `{}`,
which is trivially bisimilar).

Algorithm:
```
atoms ← {}
for each label φ:
    phi_rest ← φ
    next ← []
    for each atom a (indexed, so we can copy tail on early exit):
        inter ← a ∩ phi_rest
        if inter = ∅:
            next.append(a)
        else:
            next.append(inter)
            a_out ← a \ phi_rest
            if a_out ≠ ∅: next.append(a_out)
            phi_rest ← phi_rest \ a
            if phi_rest = ∅:
                next.append(remaining atoms unchanged   ← critical!
                break
    if phi_rest ≠ ∅: next.append(phi_rest)
    atoms ← next
```

> **Note**: The "copy remaining atoms on early break" step is essential.
> Without it, atoms that do not overlap the current label would be silently
> dropped, causing incorrect minterm sets and missed inequivalences.

**`check_equiv(a1, a2)`** — Bisimulation BFS over powerset pairs.

```
minterms ← compute_minterms(all labels from a1 and a2)

worklist ← { (initial_1, initial_2) }
while worklist not empty:
    (S1, S2) ← pop
    if visited: skip
    mark visited

    if is_final(S1) ≠ is_final(S2): return false   ← acceptance mismatch

    for each minterm μ:
        T1 ← powerset_post(S1, μ, a1)   = { q' | ∃q∈S1, q–φ–>q', μ⊆φ }
        T2 ← powerset_post(S2, μ, a2)
        enqueue (T1, T2)

return true
```

`minterm_subset_of(μ, φ)` checks `μ ∩ φ ≠ ∅` (equivalent to `μ ⊆ φ` for
true minterms, but uses `intersect` instead of `subtract` — much cheaper in
Presburger arithmetic).

---

## File Map

```
nfa.h / nfa.cpp
    NFA struct (int-letter NFA with ε-transitions)
    Thompson constructions: nfa_union, nfa_concat, nfa_star, nfa_plus
    Generic matrix types: NfaMatrix<T>, MatSR<T>
    Generic matrix ops:   mat_zero, mat_add, mat_mul, submat, mat_assemble
    matrix_star           (2×2 block formula, recursive)

regex.h / regex.cpp
    Regex AST: Zero, One, Letter, Union, Concat, Star
    Factory functions: re_zero, re_one, re_letter, re_union, re_concat,
                       re_star, re_plus  (with algebraic simplifications)
    nullable(e)          ν(e): true iff ε ∈ L(e)
    partial_deriv(e, a)  Antimirov ∂_a(e)
    regex_to_nfa(e, k)   BFS over Antimirov states → NFA

factorization.h / factorization.cpp
    Partition struct       letter → class assignment
    FacTransition          (from, RegexPtr label, to)
    FactorizationAutomaton state encoding/decoding
    build_factorization_automaton(nfa, part)
        For each class k2: build regex matrix, compute A_k2^+, emit transitions

semilinear.h / semilinear.cpp
    LinearSet, SemilinearSet
    parikh_image(e, k)   structural Parikh image of a regex
    sl_contains(S, v)    membership test (for testing)
    parikh_image_nfa(nfa, k)   Parikh image via Kleene path on semilinear matrices

parikh_mlir.h / parikh_mlir.cpp
    to_presburger_set(S, k)  SemilinearSet → MLIR PresburgerSet

symbolic_equiv.h / symbolic_equiv.cpp
    SymTransition, SymbolicAutomaton
    fac_to_symbolic(fac, k)    RegexPtr labels → PresburgerSet labels
    compute_minterms(labels, dim)
    check_equiv(a1, a2)        bisimulation BFS
```

---

## Dependency Graph

```
nfa  ◄─── regex ◄─── semilinear ◄─── parikh_mlir ◄─── symbolic_equiv
 ▲                        ▲
 └────── factorization ───┘
              ▲
          (uses regex)
```

(Arrows point from dependency to dependent.)

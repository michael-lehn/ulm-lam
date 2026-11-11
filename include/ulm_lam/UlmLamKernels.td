//===- UlmLamKernels.td - the kernel catalogue -------------*- tablegen -*-===//
//
// Declarative description of how ulm_lam expressions map onto ulm_lamk kernels.
// A custom backend (`ulm-lam-tblgen -gen-kernel-catalogue`) turns this into
//   * matchers over the normalised expression DAG used by kernel selection,
//   * the FLOP formula of each match,
//   * the builder that materialises the ulm_lamk op.
// Adding a kernel (or a vendor library) means adding records here, not
// writing C++.  Linnea's kernel description plays the same role.
//
// Patterns are written over the ulm_lam *normal form*: n-ary mul/add, and
// leaves decorated by at most one inverse and one (outer) transpose.
// A `mul` pattern with k operands matches any k *consecutive* factors of a
// longer product; this is what the matrix-chain DP enumerates.
//
//===----------------------------------------------------------------------===//

#ifndef ULM_LAM_KERNELS_TD
#define ULM_LAM_KERNELS_TD

//----------------------------------------------------------------------------
// Vocabulary
//----------------------------------------------------------------------------

// Pattern operators (ulm_lam ops in normal form).
def mul;  def add;  def trans;  def inv;  def scale;

// Constraint operators.
def has;      // (has $X, Prop...)    : X carries all properties
def vector;   // (vector $x)          : x is n x 1 (static)
def same;     // (same $A, $B)        : identical SSA value (A^T A)
def ins;      // (ins $A, $B)         : operand order of the emitted op

class Prop<string n> { string name = n; }
def General     : Prop<"general">;
def Symmetric   : Prop<"symmetric">;
def SPD         : Prop<"spd">;
def SPSD        : Prop<"spsd">;
def LowerTri    : Prop<"lower_tri">;
def UpperTri    : Prop<"upper_tri">;
def UnitDiag    : Prop<"unit_diag">;
def NonSingular : Prop<"non_singular">;
def FullRank    : Prop<"full_rank">;

// Kernel flags, spelled as in the ulm_lamk assembly format.
class Flag<string v> { string value = v; }
def Left : Flag<"left">;    def Right : Flag<"right">;
def Lower : Flag<"lower">;  def Upper : Flag<"upper">;
def N : Flag<"none">;       def T : Flag<"trans">;
def NonUnit : Flag<"non_unit">;

// The emitted op: name of the ulm_lamk op, flags, and the operand roles.
class Emit<string opName, list<Flag> flags> {
  string op = opName;
  list<Flag> flagValues = flags;
}

// A kernel: computes `pattern` into a fresh value.
//   inplace : name of the pattern variable whose buffer is overwritten
//             (bufferization inserts a copy if it is still live)
//   flops   : formula over rows()/cols() of pattern variables
//   props   : properties guaranteed for the result beyond inference
class Kernel<dag pattern, list<dag> where, Emit emit, dag operands,
             string flops, string inplace = "", list<Prop> props = []> {
  dag Pattern = pattern;
  list<dag> Where = where;
  Emit Op = emit;
  dag Operands = operands;
  string Flops = flops;
  string InPlace = inplace;
  list<Prop> Props = props;
}

// A factorisation: rewrites a leaf into a product of factors with known
// properties.  The search applies it as an *expression rewrite*
// (A -> L L^T); normal form then moves inverses onto the factors, so the
// ordinary kernel patterns (trsm, trsv) pick up the solves.
class Factorization<string name, list<dag> where, dag replacement,
                    list<dag> factorProps, Emit emit, dag operands,
                    string flops> {
  string Name = name;
  list<dag> Where = where;
  dag Replacement = replacement;
  list<dag> FactorProps = factorProps;
  Emit Op = emit;
  dag Operands = operands;
  string Flops = flops;
}

//----------------------------------------------------------------------------
// BLAS 3
//----------------------------------------------------------------------------

def : Kernel<(mul $A, $B), [], Emit<"gemm", [N, N]>, (ins $A, $B),
             "2 * rows(A) * cols(A) * cols(B)">;
def : Kernel<(mul (trans $A), $B), [], Emit<"gemm", [T, N]>, (ins $A, $B),
             "2 * cols(A) * rows(A) * cols(B)">;
def : Kernel<(mul $A, (trans $B)), [], Emit<"gemm", [N, T]>, (ins $A, $B),
             "2 * rows(A) * cols(A) * rows(B)">;

def : Kernel<(mul $S, $B), [(has $S, Symmetric)],
             Emit<"symm", [Left, Lower]>, (ins $S, $B),
             "2 * rows(S) * rows(S) * cols(B)">;

// A^T A and A A^T: half the flops of gemm, result symmetric.
def : Kernel<(mul (trans $A), $B), [(same $A, $B)],
             Emit<"syrk", [Lower, T]>, (ins $A),
             "cols(A) * cols(A) * rows(A)", "", [SPSD]>;
def : Kernel<(mul $A, (trans $B)), [(same $A, $B)],
             Emit<"syrk", [Lower, N]>, (ins $A),
             "rows(A) * rows(A) * cols(A)", "", [SPSD]>;

// Triangular solves with a matrix right-hand side.
def : Kernel<(mul (inv $L), $B), [(has $L, LowerTri)],
             Emit<"trsm", [Left, Lower, N, NonUnit]>, (ins $L, $B),
             "rows(L) * rows(L) * cols(B)", "B">;
def : Kernel<(mul (trans (inv $L)), $B), [(has $L, LowerTri)],
             Emit<"trsm", [Left, Lower, T, NonUnit]>, (ins $L, $B),
             "rows(L) * rows(L) * cols(B)", "B">;
def : Kernel<(mul (inv $U), $B), [(has $U, UpperTri)],
             Emit<"trsm", [Left, Upper, N, NonUnit]>, (ins $U, $B),
             "rows(U) * rows(U) * cols(B)", "B">;
def : Kernel<(mul $B, (inv $L)), [(has $L, LowerTri)],
             Emit<"trsm", [Right, Lower, N, NonUnit]>, (ins $L, $B),
             "rows(B) * rows(L) * rows(L)", "B">;

//----------------------------------------------------------------------------
// BLAS 2 (vector right-hand sides; more specific, so they win ties)
//----------------------------------------------------------------------------

def : Kernel<(mul $A, $x), [(vector $x)], Emit<"gemv", [N]>, (ins $A, $x),
             "2 * rows(A) * cols(A)">;
def : Kernel<(mul (trans $A), $x), [(vector $x)], Emit<"gemv", [T]>,
             (ins $A, $x), "2 * rows(A) * cols(A)">;
def : Kernel<(mul (inv $L), $x), [(has $L, LowerTri), (vector $x)],
             Emit<"trsv", [Lower, N, NonUnit]>, (ins $L, $x),
             "rows(L) * rows(L)", "x">;
def : Kernel<(mul (trans (inv $L)), $x), [(has $L, LowerTri), (vector $x)],
             Emit<"trsv", [Lower, T, NonUnit]>, (ins $L, $x),
             "rows(L) * rows(L)", "x">;

//----------------------------------------------------------------------------
// Sums of plain matrices
//----------------------------------------------------------------------------

def : Kernel<(add $Y, $X), [], Emit<"geaxpy", []>, (ins $X, $Y),
             "2 * rows(X) * cols(X)", "Y">;

//----------------------------------------------------------------------------
// Factorisations
//----------------------------------------------------------------------------

def Cholesky : Factorization<"cholesky", [(has $A, SPD)],
    (mul $L, (trans $L)), [(has $L, LowerTri, NonSingular)],
    Emit<"potrf", [Lower]>, (ins $A), "rows(A) * rows(A) * rows(A) / 3">;

// LU yields P^T L U; the permutation is applied by getrs, so LU is used
// through a dedicated solve kernel instead of a factor rewrite.
def : Kernel<(mul (inv $A), $B), [(has $A, NonSingular)],
             Emit<"getrf+getrs", [N]>, (ins $A, $B),
             "2 * rows(A) * rows(A) * rows(A) / 3"
             " + 2 * rows(A) * rows(A) * cols(B)", "B">;

#endif // ULM_LAM_KERNELS_TD

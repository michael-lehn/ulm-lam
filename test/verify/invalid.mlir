// RUN: ulm-lam-opt %s --split-input-file --verify-diagnostics

// Programs the verifiers must reject: the checks generated from the
// operation definitions (ODS (Operation Definition Specification)), the
// hand-written verify() functions, the verifier of the property encoding,
// and the comparison of a written result type with the inferred one.
// The frontends check dimensions themselves; these checks guard the IR
// (intermediate representation) whoever writes it, and every pass.

// == generated: operand constraints

func.func @integer_matrix(%a: tensor<3x3xi32>) -> tensor<3x3xi32> {
  // expected-error @below {{operand #0 must be 2D tensor of 32-bit float or 64-bit float values}}
  %t = ulm_lam.transpose %a : tensor<3x3xi32> -> tensor<3x3xi32>
  return %t : tensor<3x3xi32>
}

// -----

func.func @vector_rank1(%a: tensor<3xf64>) -> tensor<3xf64> {
  // expected-error @below {{operand #0 must be 2D tensor}}
  %t = ulm_lam.transpose %a : tensor<3xf64> -> tensor<3xf64>
  return %t : tensor<3xf64>
}

// -----

// == hand-written: ulm_lam.mul

func.func @mul_inner_dims(%a: tensor<3x4xf64>, %b: tensor<5x2xf64>) -> tensor<3x2xf64> {
  // expected-error @below {{factor 0 has 4 columns but factor 1 has 5 rows}}
  %c = ulm_lam.mul %a, %b : tensor<3x4xf64>, tensor<5x2xf64> -> tensor<3x2xf64>
  return %c : tensor<3x2xf64>
}

// -----

func.func @mul_inner_dims_later(%a: tensor<3x4xf64>, %b: tensor<4x5xf64>, %c: tensor<6x2xf64>) -> tensor<3x2xf64> {
  // expected-error @below {{factor 1 has 5 columns but factor 2 has 6 rows}}
  %p = ulm_lam.mul %a, %b, %c : tensor<3x4xf64>, tensor<4x5xf64>, tensor<6x2xf64> -> tensor<3x2xf64>
  return %p : tensor<3x2xf64>
}

// -----

func.func @mul_one_factor(%a: tensor<3x3xf64>) -> tensor<3x3xf64> {
  // expected-error @below {{needs at least two factors}}
  %c = ulm_lam.mul %a : tensor<3x3xf64> -> tensor<3x3xf64>
  return %c : tensor<3x3xf64>
}

// -----

// == hand-written: ulm_lam.add, ulm_lam.inverse, ulm_lam.lstsq, ulm_lam.assume

func.func @add_shapes(%a: tensor<3x4xf64>, %b: tensor<4x3xf64>) -> tensor<3x4xf64> {
  // expected-error @below {{terms have different shapes}}
  %c = ulm_lam.add %a, %b : tensor<3x4xf64>, tensor<4x3xf64> -> tensor<3x4xf64>
  return %c : tensor<3x4xf64>
}

// -----

func.func @inverse_not_square(%a: tensor<3x4xf64>) -> tensor<3x4xf64> {
  // expected-error @below {{operand is not square}}
  %i = ulm_lam.inverse %a : tensor<3x4xf64> -> tensor<3x4xf64>
  return %i : tensor<3x4xf64>
}

// -----

func.func @lstsq_rows(%a: tensor<5x3xf64>, %b: tensor<4x1xf64>) -> tensor<3x1xf64> {
  // expected-error @below {{A and B must have the same number of rows}}
  %x = ulm_lam.lstsq %a, %b : tensor<5x3xf64>, tensor<4x1xf64> -> tensor<3x1xf64>
  return %x : tensor<3x1xf64>
}

// -----

func.func @assume_shape(%a: tensor<3x3xf64>) -> tensor<3x4xf64> {
  // expected-error @below {{must not change shape or element type}}
  %s = ulm_lam.assume %a : tensor<3x3xf64> -> tensor<3x4xf64>
  return %s : tensor<3x4xf64>
}

// -----

// == the property encoding (verifyEncoding)

// expected-error @below {{zero contradicts full rank}}
func.func @zero_full_rank(%a: tensor<3x3xf64, #ulm_lam.props<zero, full_rank>>) {
  return
}

// -----

// expected-error @below {{square property on a 3x4 tensor}}
func.func @square_3x4(%a: tensor<3x4xf64, #ulm_lam.props<spd>>) {
  return
}

// -----

// expected-error @below {{lower_stored and upper_stored exclude each other}}
func.func @both_triangles(%a: tensor<3x3xf64, #ulm_lam.props<symmetric, lower_stored, upper_stored>>) {
  return
}

// -----

// expected-error @below {{need a symmetric matrix}}
func.func @stored_not_symmetric(%a: tensor<3x3xf64, #ulm_lam.props<lower_stored>>) {
  return
}

// -----

// == result types must be justified by inference

func.func @spd_times_spd(%a: tensor<3x3xf64, #ulm_lam.props<spd>>, %b: tensor<3x3xf64, #ulm_lam.props<spd>>) -> tensor<3x3xf64, #ulm_lam.props<spd>> {
  // expected-error @below {{inferred type(s)}}
  // expected-error @below {{failed to infer returned types}}
  %c = ulm_lam.mul %a, %b : tensor<3x3xf64, #ulm_lam.props<spd>>, tensor<3x3xf64, #ulm_lam.props<spd>> -> tensor<3x3xf64, #ulm_lam.props<spd>>
  return %c : tensor<3x3xf64, #ulm_lam.props<spd>>
}

// -----

func.func @mul_result_shape(%a: tensor<3x4xf64>, %b: tensor<4x2xf64>) -> tensor<3x3xf64> {
  // expected-error @below {{inferred type(s)}}
  // expected-error @below {{failed to infer returned types}}
  %c = ulm_lam.mul %a, %b : tensor<3x4xf64>, tensor<4x2xf64> -> tensor<3x3xf64>
  return %c : tensor<3x3xf64>
}

// -----

// == kernels (ulm_lamk): shapes of BLAS calls

func.func @gemm_shapes(%a: tensor<3x4xf64>, %b: tensor<5x2xf64>, %c: tensor<3x2xf64>) -> tensor<3x2xf64> {
  %one = arith.constant 1.0 : f64
  // expected-error @below {{shapes of op(A), op(B) and C do not match}}
  %r = ulm_lamk.gemm trans_a(none) trans_b(none) alpha(%one) beta(%one) ins(%a, %b : tensor<3x4xf64>, tensor<5x2xf64>) outs(%c : tensor<3x2xf64>) -> tensor<3x2xf64>
  return %r : tensor<3x2xf64>
}

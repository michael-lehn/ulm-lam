// RUN: ulm-lam --backend=cblas --emit=mojo %s -o - | FileCheck %s
// RUN: ulm-lam --backend=cblas --emit=mojo --mojo-c-abi %s -o - | FileCheck %s --check-prefix=CABI
//
// ulm-lam --emit=mojo (tools/ulm-lam.cpp, MojoEmitter): the example of the
// lectures, x = A^-1 b for sensor data b in every step.  The LU (lower-upper)
// factorization once before the loop (getrf), in the loop only the solve
// on b in place (getrs); the caller's functions as the methods of the
// trait Externs, with --mojo-c-abi as C functions and the program exported.

func.func private @get_sensor_data(index, index) -> (tensor<?x1xf64> {ulm_lam.dims = [0]})
func.func private @react(index, index, tensor<?x1xf64>)

func.func @control(%n: index, %A: tensor<?x?xf64> {ulm_lam.in, ulm_lam.dims = [0, 0]}, %steps: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %l = %c0 to %steps step %c1 {
    %b = func.call @get_sensor_data(%n, %l) : (index, index) -> tensor<?x1xf64>
    %Ai = ulm_lam.inverse %A : tensor<?x?xf64> -> tensor<?x?xf64>
    %x = ulm_lam.mul %Ai, %b : tensor<?x?xf64>, tensor<?x1xf64> -> tensor<?x1xf64>
    func.call @react(%n, %l, %x) : (index, index, tensor<?x1xf64>) -> ()
  }
  return
}

// CHECK: from ulm_lam import kernels as K
// CHECK: trait Externs:
// CHECK: def get_sensor_data(mut self, a1: Int, a2: Int, a3: P[DType.float64]): ...
// CHECK: def react(mut self, a1: Int, a2: Int, a3: P[DType.float64]): ...
// CHECK: # control: matrices by reference (row-major), owned by the caller
// CHECK: def control[E: Externs](mut ext: E, [[N:v[0-9]+]]: Int, [[A:v[0-9]+]]: P[DType.float64], [[S:v[0-9]+]]: Int, {{.*}}) -> Int:
// CHECK: K.copy({{.*}}, [[A]],
// CHECK: = K.getrf([[N]], [[N]],
// CHECK: for {{v[0-9]+}} in range({{v[0-9]+}}, [[S]], {{v[0-9]+}}):
// CHECK-NEXT: ext.get_sensor_data([[N]],
// CHECK-NEXT: _ = K.getrs(False, False, [[N]], 1,
// CHECK-NEXT: ext.react([[N]],
// CHECK-NOT: @export

// CABI: from std.ffi import external_call
// CABI: struct CExterns(Externs):
// CABI: def get_sensor_data(mut self, a1: Int, a2: Int, a3: P[DType.float64]):
// CABI-NEXT: external_call["get_sensor_data", NoneType](a1, a2, a3)
// CABI: @export("control")
// CABI-NEXT: def control_c_abi({{.*}}) abi("C") -> Int:
// CABI-NEXT: var ext = CExterns()
// CABI-NEXT: return control(ext,

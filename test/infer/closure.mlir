// RUN: ulm-lam-opt %s --ulm-lam-test-closure 2>&1 | FileCheck %s

// The implication closure of matrix properties (closure() in
// lib/UlmLamDialect.cpp) is complete for every set of property bits:
// applying it a second time adds nothing.  Checked exhaustively, with an
// unknown shape and with a static square one (2 x 2^15 sets).

// CHECK: remark: closure: 65536 property sets, each closure is a fixpoint
module {
}

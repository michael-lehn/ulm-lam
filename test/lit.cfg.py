# lit configuration of ulm.lam's pass tests:  make check
#
# Every test is a .mlir file with RUN lines (ulm-lam-opt ... | FileCheck).
# The tools: ulm-lam-opt from build/bin, FileCheck and not from LLVM
# (LLVM_PREFIX, or Homebrew's llvm).
import os
import shutil
import subprocess

import lit.formats

config.name = "ulm.lam"
config.test_format = lit.formats.ShTest()  # (the internal shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(os.path.dirname(__file__), "..", "build",
                                     "test")

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
llvm = os.environ.get("LLVM_PREFIX")
if not llvm and shutil.which("brew"):
    llvm = subprocess.run(["brew", "--prefix", "llvm"], capture_output=True,
                          text=True).stdout.strip()
paths = [os.path.join(root, "build", "bin")]
if llvm:
    paths.append(os.path.join(llvm, "bin"))
config.environment["PATH"] = os.pathsep.join(
    paths + [config.environment.get("PATH", os.environ.get("PATH", ""))])

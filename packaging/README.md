# Registering UlmLAM.jl

Two packages go into Julia's General registry:

1. **ulm_lam_jll**: the compiler executable `ulm-lam` for every platform
   (Linux, macOS, Windows, FreeBSD; 64-bit), built by Yggdrasil's
   BinaryBuilder from [`yggdrasil/U/ulm_lam/build_tarballs.jl`](yggdrasil/U/ulm_lam/build_tarballs.jl).
   It links MLIR (Multi-Level Intermediate Representation) and LLVM (Low
   Level Virtual Machine) statically, so it works under any Julia version.
2. **UlmLAM**: the Julia package in [`julia/UlmLAM`](../julia/UlmLAM), which
   then depends on ulm_lam_jll.

What is verified: the CMake build with static LLVM against the artifact of
LLVM_full_jll 23.1.4 on x86_64 Linux and the tests with the result
(workflow `jll-check`), the tests on Linux and macOS with Julia 1.10 and 1
(workflow `CI`).  Not verified: the cross-compiled builds of BinaryBuilder
for the other platforms; Yggdrasil's CI builds them all for the pull
request.

## Step 1: ulm_lam_jll (pull request to Yggdrasil)

```sh
gh repo fork JuliaPackaging/Yggdrasil --clone && cd Yggdrasil
git switch -c ulm_lam
mkdir -p U/ulm_lam
cp ../ulm-lam/packaging/yggdrasil/U/ulm_lam/build_tarballs.jl U/ulm_lam/
git add U/ulm_lam && git commit -m "[ulm_lam] new package: the ulm.lam compiler"
git push -u origin ulm_lam
gh pr create --repo JuliaPackaging/Yggdrasil --title "[ulm_lam] new package" \
  --body "The compiler of ulm.lam (https://github.com/michael-lehn/ulm-lam): linear algebra expressions onto BLAS and LAPACK, via MLIR. One executable, MLIR and LLVM linked statically from LLVM_full_jll 23.1.4."
```

Yggdrasil's CI builds every platform.  Failures on single platforms can
be answered by restricting `platforms` in the recipe (e.g. Windows or
FreeBSD); the Yggdrasil maintainers usually help.  After the merge,
ulm_lam_jll is registered automatically (a few hours).

## Step 2: UlmLAM depends on ulm_lam_jll

Once ulm_lam_jll is registered (this step can be done here):

```sh
julia --project=julia/UlmLAM -e 'using Pkg; Pkg.add("ulm_lam_jll")'
```

add `ulm_lam_jll = "0.1"` to `[compat]`, and in
`julia/UlmLAM/src/runtime.jl` let `jll_tool()` return
`ulm_lam_jll.ulm_lam_path` (with `import ulm_lam_jll` in
`src/UlmLAM.jl`).  The CI workflow then needs no LLVM any more.

## Step 3: UlmLAM (Registrator)

Install the [Registrator app](https://github.com/JuliaRegistries/Registrator.jl)
for the repository (once), then comment on the commit to register (on
GitHub, in the commit's page):

```
@JuliaRegistrator register subdir=julia/UlmLAM
```

New packages wait three days in General; the AutoMerge bot checks the
name, the `[compat]` entries and that the package loads.  After the merge,
TagBot (`.github/workflows/TagBot.yml`) tags the release, and
`] add UlmLAM` works for everyone.

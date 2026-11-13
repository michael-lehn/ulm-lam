# ULMBLAS_DIR (default ~/git/ulmBLAS) and ARCHFLAGS (default -mcpu=native)
# select ulmBLAS and its micro kernels, see scripts/ulmblas-env.sh.
.PHONY: all ods plugin tool check egraph egraph-ir egraph-fuzz calls stages loop gls kalman compose cases dynamic gp showcases torch python julia doc clean

all: ods plugin check stages loop gls kalman compose calls cases dynamic python julia doc

ods:            ## run all mlir-tblgen backends on the dialect sketch
	scripts/check-ods.sh

plugin:         ## build the mlir-opt plugin: dialects + passes (lib/*.cpp)
	scripts/build-plugin.sh

tool:           ## build ulm-lam-opt (tools/ulm-lam-opt.cpp): mlir-opt with ulm.lam's dialects and passes
	scripts/build-tool.sh

check: tool     ## the pass tests in test/ (lit, FileCheck; lit from PyPI into build/venv-lit)
	@test -x build/venv-lit/bin/lit || (python3 -m venv build/venv-lit && build/venv-lit/bin/pip install -q lit)
	build/venv-lit/bin/lit -sv test

egraph:         ## equality saturation experiment (egglog, experiments/egraph; Python 3.12)
	@test -x build/venv-egglog/bin/python || (python3.12 -m venv build/venv-egglog && build/venv-egglog/bin/pip install -q egglog numpy scipy)
	build/venv-egglog/bin/python experiments/egraph/lamp_egraph.py

egraph-ir: tool  ## e-graph vs. ulm.lam's selection on problems traced from the Python frontend
	@test -x build/venv-egglog/bin/python || (python3.12 -m venv build/venv-egglog && build/venv-egglog/bin/pip install -q egglog numpy scipy)
	build/venv-egglog/bin/python experiments/egraph/lamp_ir.py

egraph-fuzz: tool ## random problems: ulm.lam's selection against the e-graph (optimal extraction)
	@test -x build/venv-egglog/bin/python || (python3.12 -m venv build/venv-egglog && build/venv-egglog/bin/pip install -q egglog numpy scipy)
	build/venv-egglog/bin/python experiments/egraph/lamp_fuzz.py 200 1

stages:         ## GLS stages 1-4 parse/verify; stage 3 bufferizes as stage 4 predicts
	scripts/check-stages.sh

loop:           ## X += A*B*C in a loop: selection, memory, C++ with ulmBLAS, run
	examples/loop/check.sh
	examples/loop/run.sh

gls:            ## GLS from its ulm.lam normal form: derivation, C++ with ulmBLAS, run
	examples/gls/check.sh
	examples/gls/run.sh

kalman:         ## formula (1) in a loop with changing y: hoisting, C++ with ulmBLAS, run
	examples/kalman/check.sh
	examples/kalman/run.sh

calls:          ## functions calling functions (examples/calls): Julia and Python, the same IR
	examples/calls/run.sh

compose:        ## functions with workspaces calling such functions: accumulated queries
	examples/compose/run.sh

cases:          ## inverses, solves and sums: C++ with ulmBLAS against naive code
	examples/cases/run.sh

dynamic:        ## GLS with sizes as run-time parameters: one library, several sizes
	examples/dynamic/run.sh

gp:             ## Gaussian process regression on public data (downloaded once)
	python3 python/examples/gp.py

showcases:      ## ridge, linear attention, Kalman filter, Bayesian regression (Python vs NumPy)
	python3 python/examples/ridge.py
	python3 python/examples/attention.py
	python3 python/examples/kalman_filter.py
	python3 python/examples/bayes.py

torch:          ## a linear layer trained as formulas against PyTorch (venv build/venv-torch)
	test -x build/venv-torch/bin/python || (python3 -m venv build/venv-torch && \
		build/venv-torch/bin/pip install --quiet numpy torch)
	ULM_LAM_BLAS_LIBS="-framework Accelerate" OPENMP=0 \
		build/venv-torch/bin/python python/examples/torch_linear.py

python:         ## the Python frontend: unit tests and the Kalman examples
	python3 -m unittest discover -s python/tests
	python3 python/examples/kalman.py
	python3 python/examples/adaptive_kalman.py

julia:          ## the Julia frontend (julia/UlmLam): tests and the showcases in Julia
	julia --project=julia/UlmLam julia/UlmLam/test/runtests.jl
	julia --project=julia/UlmLam julia/examples/kalman.jl
	julia --project=julia/UlmLam julia/examples/gp.jl
	julia --project=julia/UlmLam julia/examples/ridge.jl
	julia --project=julia/UlmLam julia/examples/attention.jl
	julia --project=julia/UlmLam julia/examples/kalman_filter.jl
	julia --project=julia/UlmLam julia/examples/bayes.jl

doc:            ## build doc/ulm-lam.pdf and the Field Guide companion with LuaLaTeX
	cd doc && latexmk
	cd doc/field-guide-companion && latexmk

clean:
	rm -rf build
	cd doc && latexmk -C
	cd doc/field-guide-companion && latexmk -C

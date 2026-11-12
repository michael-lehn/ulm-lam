# ULMBLAS_DIR (default ~/git/ulmBLAS) and ARCHFLAGS (default -mcpu=native)
# select ulmBLAS and its micro kernels, see scripts/ulmblas-env.sh.
.PHONY: all ods plugin stages loop gls kalman compose cases dynamic gp python doc clean

all: ods plugin stages loop gls kalman compose cases dynamic python doc

ods:            ## run all mlir-tblgen backends on the dialect sketch
	scripts/check-ods.sh

plugin:         ## build the mlir-opt plugin: dialects + passes (lib/*.cpp)
	scripts/build-plugin.sh

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

compose:        ## functions with workspaces calling such functions: accumulated queries
	examples/compose/run.sh

cases:          ## inverses, solves and sums: C++ with ulmBLAS against naive code
	examples/cases/run.sh

dynamic:        ## GLS with sizes as run-time parameters: one library, several sizes
	examples/dynamic/run.sh

gp:             ## Gaussian process regression on public data (downloaded once)
	python3 python/examples/gp.py

python:         ## the Python frontend: unit tests and the Kalman examples
	python3 -m unittest discover -s python/tests
	python3 python/examples/kalman.py
	python3 python/examples/adaptive_kalman.py

doc:            ## build doc/ulm-lam.pdf with LuaLaTeX
	cd doc && latexmk

clean:
	rm -rf build
	cd doc && latexmk -C

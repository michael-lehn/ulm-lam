# Run the notebook without a browser and print errors and outputs:
#     julia --project=notebook notebook/run_headless.jl notebook/ulm-lam.jl
using Pluto
path = ARGS[1]
s = Pluto.ServerSession()
s.options.server.disable_writing_notebook_files = true
t = @elapsed nb = Pluto.SessionActions.open(s, path; run_async=false)
println("ran in $(round(t, digits=1)) s")
for c in nb.cells
    if c.errored
        println("ERROR in cell: ", first(c.code, 80)); println(c.output.body)
    end
end
for c in nb.cells
    if !startswith(c.code, "md") && c.output.body isa String
        println("---- ", first(replace(c.code, "\n" => " "), 70)); println(first(c.output.body, 1500))
    end
end
Pluto.SessionActions.shutdown(s, nb)

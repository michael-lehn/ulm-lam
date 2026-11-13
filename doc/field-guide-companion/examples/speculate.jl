using UlmLAM, LinearAlgebra
@lam function iterate(M::Mat(:n, :n, :spd), x0::Vec(:n), steps::Index)
    loop(steps, x0) do i, x
        (inv(M) * x,)
    end
end
Mbad = [1.0 2.0; 2.0 1.0]   # symmetric, indefinite
x0 = [1.0, 1.0]
for steps in (0, 1)
    try
        println("steps = $steps: ", iterate(Mbad, x0, steps))
    catch e
        print("steps = $steps: "); showerror(stdout, e); println()
    end
end

using UlmLAM

const camera = extern(:camera, [Index], Vec(:m))       # the image of frame i
const steer = extern(:steer, [Index, Vec(:k)])         # the motor commands

@lam function fly(W::Mat(:n, :m), A::Mat(:n, :n, :spd), K::Mat(:k, :n), frames::Index)
    loop(frames) do i
        image = camera(i)
        f = W * image          # features of the image: a linear layer
        x = inv(A) * f         # the estimated state of the drone
        steer(i, K * x)        # the controller
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    println(code_lam(fly))
end

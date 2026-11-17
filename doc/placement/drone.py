from ulm import lam
from ulm.lam import Matrix, Vector, Index

camera = lam.extern("camera", [Index], Vector("m"))       # the image of frame t
steer = lam.extern("steer", [Index, Vector("k")])         # the motor commands


@lam.function
def fly(W: Matrix("n", "m"), A: Matrix("n", "n", "spd"), K: Matrix("k", "n"),
        frames: Index):
    with lam.loop(frames) as L:
        image = camera(L.index)
        f = W @ image                # features of the image: a linear layer
        x = A.inv @ f                # the estimated state of the drone
        steer(L.index, K @ x)        # the controller


if __name__ == "__main__":
    print(lam.module([fly]), end="")

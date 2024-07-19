import numpy as np


def camera_params():
    # RGB Intrinsic Parameters
    fx = 5.1885790117450188e+02
    fy = 5.1946961112127485e+02
    cx = 3.2558244941119034e+02
    cy = 2.5373616633400465e+02

    # RGB Distortion Parameters
    k1 =  2.0796615318809061e-01
    k2 = -5.8613825163911781e-01
    p1 = 7.2231363135888329e-04
    p2 = 1.0479627195765181e-03
    k3 = 4.9856986684705107e-01

    return (fx, fy, cx, cy), (k1, k2, p1, p2, k3)


def make_K(fx, fy, cx, cy):
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K

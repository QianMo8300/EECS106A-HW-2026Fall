"""Student functions for HW6: trajectories and integration in SE(3)."""

import numpy as np

from hw2.kin_func_skeleton import inverse_SE3, renormalize_SE3, se3_to_twist, twist_to_SE3


L1 = 0.57
L2 = 0.25
L3 = 0.20
L4 = 0.10
K = 0.5


def trajectory(t):
    """Return the chair-swing pose defined in the handout.

    Args:
        t: Scalar time in seconds, t >= 0.

    Returns:
        NumPy array with shape (4, 4), representing g_ab(t) in the ride
        frame.
    """
    # TODO
    phi = K * np.sin(t)
    st, ct = np.sin(t), np.cos(t)
    sp, cp = np.sin(phi), np.cos(phi)

    p = np.array([
        -L2 * st,
        L2 * ct,
        L1 + L4 - L4 * ct,
    ]) + np.array([
        -L3 * st * sp,
        L3 * ct * sp,
        -L3 * cp,
    ])

    Bx = np.array([-ct, -st, 0.0])
    Bz = np.array([sp * st, -sp * ct, cp])
    # By = -np.array([cp * st, -cp * ct, sp]) 
    By = np.cross(Bz, Bx)
    R = np.column_stack((Bx, By, Bz))

    g = np.eye(4)
    g[:3, :3] = R
    g[:3, 3] = p
    return g


def body_velocity(t):
    """Calculate the instantaneous body velocity of the chair swing.

    Args:
        t: Scalar time in seconds, t >= 0.

    Returns:
        NumPy array with shape (4, 4), representing the body velocity
        matrix in se(3). Analytically differentiate the full trajectory
        matrix, including its rotation, and compute the body velocity.

    inverse_SE3 from kin_func_skeleton inverts the pose.
    """
    # TODO
    phi = K * np.sin(t)
    phid = K * np.cos(t)                  
    st, ct = np.sin(t), np.cos(t)
    sp, cp = np.sin(phi), np.cos(phi)

    pdot = np.array([
        -L2 * ct,
        -L2 * st,
        L4 * st,
    ]) + np.array([
        -L3 * ct * sp - L3 * st * cp * phid,
        -L3 * st * sp + L3 * ct * cp * phid,
        L3 * sp * phid,
    ])

    dBx = np.array([st, -ct, 0.0])
    dBy = np.array([
        ct * cp - st * sp * phid,
        st * cp + ct * sp * phid,
        -cp * phid,
    ])

    dBz = np.array([
        ct * sp + st * cp * phid,
        st * sp - ct * cp * phid,
        -sp * phid,
    ])
    # dBy = np.cross(dBz, Bx) + np.cross(Bz, dBx)
    Rdot = np.column_stack((dBx, dBy, dBz))

    gdot = np.zeros((4, 4))
    gdot[:3, :3] = Rdot
    gdot[:3, 3] = pdot

    return inverse_SE3(trajectory(t)) @ gdot


def euler_step(pose, velocity_hat, dt):
    """Advance an SE(3) pose with body velocity held constant for one step.

    Args:
        pose: NumPy array with shape (4, 4), the current pose.
        velocity_hat: NumPy array with shape (4, 4), a body velocity in se(3).
        dt: Nonnegative scalar timestep in seconds.

    Returns:
        NumPy array with shape (4, 4), the next SE(3) pose. Apply the
        exponential rigid-motion increment on the right of the current
        pose, as in Problem 4.

    se3_to_twist converts the velocity matrix to a vector. twist_to_SE3
    computes the rigid-motion increment over dt. Apply renormalize_SE3
    to the updated pose to correct floating-point drift in its rotation.
    """
    # TODO
    xi = se3_to_twist(velocity_hat)        
    increment = twist_to_SE3(xi, dt)  
    new_pose = pose @ increment     
    return renormalize_SE3(new_pose)
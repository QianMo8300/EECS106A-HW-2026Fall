"""Functions to help with Jacobian calculations."""

import numpy as np
from scipy.linalg import logm

from hw2.kin_func_skeleton import (
    R3_to_so3,
    forward_kinematics,
    inverse_SE3,
    se3_to_twist,
    twist_to_SE3,
)


def adjoint(g):
    """Calculate the adjoint matrix of a rigid transformation.

    Args:
        g: NumPy array with shape ``(4, 4)`` representing a
            transformation in SE(3).

    Returns:
        NumPy array with shape ``(6, 6)`` representing ``Ad_g``.
    """

    # TODO
    R = g[:3, :3]
    p = g[:3, 3]

    p_hat = np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ])

    Ad_g = np.zeros((6, 6))
    Ad_g[:3, :3] = R
    Ad_g[:3, 3:] = p_hat @ R
    Ad_g[3:, 3:] = R
    return Ad_g


def adjoint_inv(g):
    """Calculate the inverse adjoint matrix of a rigid transformation.

    Args:
        g: NumPy array with shape ``(4, 4)`` representing a
            transformation in SE(3).

    Returns:
        NumPy array with shape ``(6, 6)`` representing ``Ad_g^{-1}``.
    """

    # TODO
    R = g[:3, :3]
    p = g[:3, 3]

    p_hat = np.array([
        [0, -p[2], p[1]],
        [p[2], 0, -p[0]],
        [-p[1], p[0], 0]
    ])

    Ad_g_inv = np.zeros((6, 6))
    Ad_g_inv[:3, :3] = R.T
    Ad_g_inv[:3, 3:] = -R.T @ p_hat
    Ad_g_inv[3:, 3:] = R.T
    return Ad_g_inv


def spatial_jacobian(twists, theta):
    """Calculate the spatial manipulator Jacobian.

    Args:
        twists: NumPy array with shape ``(6, N)`` whose columns are the
            spatial twists of the joints at the home configuration.
        theta: NumPy array with shape ``(N,)`` containing the joint values.

    Returns:
        NumPy array with shape ``(6, N)`` containing the spatial Jacobian at
        ``theta``.
    """

    # TODO
    n = twists.shape[1]
    J = np.zeros((6, n))
    g = np.eye(4)

    for i in range(n):
        J[:, i] = adjoint(g) @ twists[:, i]

        g_i = twist_to_SE3(twists[:, i], theta[i])
        g = g @ g_i

    return J


def body_jacobian(twists, theta, g_st0):
    """Calculate the body manipulator Jacobian.

    Args:
        twists: NumPy array with shape ``(6, N)`` whose columns are the
            spatial twists of the joints at the home configuration.
        theta: NumPy array with shape ``(N,)`` containing the joint values.
        g_st0: NumPy array with shape ``(4, 4)`` containing the end-effector
            pose at the home configuration.

    Returns:
        NumPy array with shape ``(6, N)`` containing the body Jacobian at
        ``theta``.
    """

    # TODO
    J_s = spatial_jacobian(twists, theta)

    g_st = forward_kinematics(twists, theta) @ g_st0

    J_b = adjoint_inv(g_st) @ J_s
    return J_b


def spatial_velocity(twists, theta, theta_dot):
    """Calculate the spatial end-effector velocity.

    Args:
        twists: NumPy array with shape ``(6, N)`` whose columns are the
            spatial twists of the joints at the home configuration.
        theta: NumPy array with shape ``(N,)`` containing the joint values.
        theta_dot: NumPy array with shape ``(N,)`` containing the joint
            velocities.

    Returns:
        NumPy array with shape ``(6,)`` containing the spatial velocity.
    """

    # TODO
    J_s = spatial_jacobian(twists, theta)
    V_s = J_s @ theta_dot
    return V_s


def body_velocity(twists, theta, theta_dot, g_st0):
    """Calculate the body end-effector velocity.

    Args:
        twists: NumPy array with shape ``(6, N)`` whose columns are the
            spatial twists of the joints at the home configuration.
        theta: NumPy array with shape ``(N,)`` containing the joint values.
        theta_dot: NumPy array with shape ``(N,)`` containing the joint
            velocities.
        g_st0: NumPy array with shape ``(4, 4)`` containing the end-effector
            pose at the home configuration.

    Returns:
        NumPy array with shape ``(6,)`` containing the body velocity.
    """

    # TODO
    J_b = body_jacobian(twists, theta, g_st0)
    V_b = J_b @ theta_dot
    return V_b


def spatial_configuration_error(pose, desired_pose):
    """Calculate the spatial-frame transformation error.

    Args:
        pose: NumPy array with shape ``(4, 4)`` for the current SE(3) pose.
        desired_pose: NumPy array with shape ``(4, 4)`` for the desired
            SE(3) pose, expressed in the same reference frame.

    Returns:
        NumPy array with shape ``(4, 4)`` containing our error.
    """

    # TODO
    g_E = desired_pose @ inverse_SE3(pose)

    return g_E


def spatial_error_twist(error):
    """Convert a transformation error into an exponential-coordinate vector.

    Args:
        error: NumPy array with shape ``(4, 4)`` in SE(3), returned by
            ``spatial_configuration_error``. Assume its rotation angle is less than pi.

    Returns:
        NumPy array with shape ``(6,)`` containing ``log(error)^vee``.

    Use ``scipy.linalg.logm`` for the matrix logarithm.
    """

    # TODO
    error_hat = logm(error)
    xi_e = se3_to_twist(error_hat)
    return np.real_if_close(xi_e)


def jacobian_inverse_kinematics(
    twists, theta, g_st0, desired_pose, step_size=0.2,
    max_iterations=200, tolerance=1e-6,
):
    """Solve inverse kinematics using spatial errors and forward-Euler updates.

    Args:
        twists: NumPy array with shape ``(6, N)`` containing the joint twists
            at the home configuration.
        theta: NumPy array with shape ``(N,)`` containing the initial joint values.
        g_st0: NumPy array with shape ``(4, 4)`` containing the home tool pose.
        desired_pose: NumPy array with shape ``(4, 4)`` containing the target
            tool pose in the same reference frame.
        step_size: Nonnegative scalar multiplier for each joint update.
        max_iterations: Nonnegative integer limiting the number of updates.
        tolerance: Positive scalar threshold for the norm of the spatial
            error twist. Assume relative rotation angles are less than pi.

    Returns:
        ``(theta, history)``, where theta is a NumPy array with shape ``(N,)``
        and history has shape ``(M, N)``. Include the initial joint values
        and every update, so ``1 <= M <= max_iterations + 1`` and
        ``theta == history[-1]``. Return the latest values even if the target
        is not reached.
    """

    # TODO
    theta_current = np.array(theta, dtype=float, copy=True)
    history = [theta_current.copy()]

    for _ in range(max_iterations):
        pose = forward_kinematics(twists, theta_current) @ g_st0
        error = spatial_configuration_error(pose, desired_pose)
        xi_e = spatial_error_twist(error)

        if np.linalg.norm(xi_e) <= tolerance:
            break

        J_s = spatial_jacobian(twists, theta_current)
        theta_current += step_size * (np.linalg.pinv(J_s) @ xi_e)

        history.append(theta_current.copy())

    return theta_current, np.array(history)

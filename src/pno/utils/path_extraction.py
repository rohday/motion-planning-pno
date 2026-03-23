"""Path extraction via gradient descent on a value field."""

import numpy as np
import torch


def compute_gradient(value_field, mask=None):
    """Compute finite-difference gradient, optionally masked by free space."""
    H, W = value_field.shape

    if mask is not None:
        V = value_field.copy()
        V[mask < 0.5] = np.nan
    else:
        V = value_field

    grad_y = np.zeros((H, W))
    grad_x = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            if mask is not None and mask[i, j] < 0.5:
                continue

            has_up = (i > 0) and (mask is None or mask[i-1, j] > 0.5)
            has_down = (i < H-1) and (mask is None or mask[i+1, j] > 0.5)

            if has_up and has_down:
                grad_y[i, j] = (V[i+1, j] - V[i-1, j]) / 2.0
            elif has_down:
                grad_y[i, j] = V[i+1, j] - V[i, j]
            elif has_up:
                grad_y[i, j] = V[i, j] - V[i-1, j]

            has_left = (j > 0) and (mask is None or mask[i, j-1] > 0.5)
            has_right = (j < W-1) and (mask is None or mask[i, j+1] > 0.5)

            if has_left and has_right:
                grad_x[i, j] = (V[i, j+1] - V[i, j-1]) / 2.0
            elif has_right:
                grad_x[i, j] = V[i, j+1] - V[i, j]
            elif has_left:
                grad_x[i, j] = V[i, j] - V[i, j-1]

    return grad_y, grad_x


def interpolate_gradient(grad_y, grad_x, y, x):
    """Bilinear interpolation of gradient at sub-pixel position (y, x)."""
    H, W = grad_y.shape
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = min(y0 + 1, H - 1)
    x1 = min(x0 + 1, W - 1)
    y0 = max(y0, 0)
    x0 = max(x0, 0)

    fy = y - y0
    fx = x - x0

    w00 = (1 - fy) * (1 - fx)
    w01 = (1 - fy) * fx
    w10 = fy * (1 - fx)
    w11 = fy * fx

    gy = w00 * grad_y[y0, x0] + w01 * grad_y[y0, x1] + \
         w10 * grad_y[y1, x0] + w11 * grad_y[y1, x1]
    gx = w00 * grad_x[y0, x0] + w01 * grad_x[y0, x1] + \
         w10 * grad_x[y1, x0] + w11 * grad_x[y1, x1]

    return gy, gx


def extract_path(value_field, start, goal, mask,
                 step_size=0.8, max_steps=500, goal_threshold=1.5):
    """Extract a path from `start` to `goal` by descending the field."""
    H, W = value_field.shape
    grad_y, grad_x = compute_gradient(value_field, mask)

    path = [np.array(start, dtype=np.float64)]
    y, x = float(start[0]), float(start[1])

    for step in range(max_steps):
        dist_to_goal = np.sqrt((y - goal[0])**2 + (x - goal[1])**2)
        if dist_to_goal < goal_threshold:
            path.append(np.array(goal, dtype=np.float64))
            return path, True

        gy, gx = interpolate_gradient(grad_y, grad_x, y, x)

        grad_norm = np.sqrt(gy**2 + gx**2)
        if grad_norm < 1e-6:
            break

        gy /= grad_norm
        gx /= grad_norm

        new_y = y - step_size * gy
        new_x = x - step_size * gx

        new_y = np.clip(new_y, 0.5, H - 1.5)
        new_x = np.clip(new_x, 0.5, W - 1.5)

        iy, ix = int(round(new_y)), int(round(new_x))
        iy = np.clip(iy, 0, H - 1)
        ix = np.clip(ix, 0, W - 1)

        if mask[iy, ix] < 0.5:
            # Try smaller step
            new_y = y - 0.3 * step_size * gy
            new_x = x - 0.3 * step_size * gx
            iy, ix = int(round(new_y)), int(round(new_x))
            iy = np.clip(iy, 0, H - 1)
            ix = np.clip(ix, 0, W - 1)
            if mask[iy, ix] < 0.5:
                break

        y, x = new_y, new_x
        path.append(np.array([y, x]))

        if len(path) > 5:
            d_3back = np.linalg.norm(path[-1] - path[-4])
            if d_3back < step_size * 0.5:
                break

    return path, False

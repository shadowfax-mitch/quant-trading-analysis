from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


@dataclass
class OUParams:
    mu: float
    theta: float
    sigma: float
    a: float
    b: float
    sigma_eps: float


def estimate_ou_params(prices: Iterable[float], dt: float = 1.0) -> Optional[OUParams]:
    values = np.asarray(list(prices), dtype=float)
    if values.size < 3:
        return None

    x = values[:-1]
    y = values[1:]
    n = x.size

    sum_x = float(x.sum())
    sum_y = float(y.sum())
    sum_xx = float(np.dot(x, x))
    sum_yy = float(np.dot(y, y))
    sum_xy = float(np.dot(x, y))

    denom = n * sum_xx - sum_x * sum_x
    if denom <= 0:
        return None

    b = (n * sum_xy - sum_x * sum_y) / denom
    if not (0 < b < 1):
        return None

    a = (sum_y - b * sum_x) / n
    # SSE from sums to avoid per-point residual loops.
    sse = (
        sum_yy
        + a * a * n
        + b * b * sum_xx
        + 2 * a * b * sum_x
        - 2 * a * sum_y
        - 2 * b * sum_xy
    )
    if n <= 2:
        return None
    sigma_eps2 = sse / (n - 2)
    if sigma_eps2 <= 0:
        return None

    theta = -np.log(b) / dt
    if theta <= 0:
        return None

    denom_sigma = 1 - b * b
    if denom_sigma <= 0:
        return None

    sigma = np.sqrt(sigma_eps2 * 2 * theta / denom_sigma)
    if sigma <= 0:
        return None

    mu = a / (1 - b)
    return OUParams(mu=mu, theta=theta, sigma=sigma, a=a, b=b, sigma_eps=float(np.sqrt(sigma_eps2)))


def _forward_fill(values: np.ndarray) -> None:
    last = np.nan
    for i in range(values.size):
        if np.isnan(values[i]):
            values[i] = last
        else:
            last = values[i]


def rolling_ou_params(
    prices: Iterable[float],
    window: int,
    dt: float = 1.0,
    lag: int = 1,
    step: int = 1,
    forward_fill: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(list(prices), dtype=float)
    size = values.size
    mu = np.full(size, np.nan)
    theta = np.full(size, np.nan)
    sigma = np.full(size, np.nan)

    if size < window + 1:
        return mu, theta, sigma

    if window < 3:
        return mu, theta, sigma

    if step < 1:
        raise ValueError("step must be >= 1")

    x = values[:-1]
    y = values[1:]
    m = x.size

    csum_x = np.concatenate([[0.0], np.cumsum(x)])
    csum_y = np.concatenate([[0.0], np.cumsum(y)])
    csum_xx = np.concatenate([[0.0], np.cumsum(x * x)])
    csum_yy = np.concatenate([[0.0], np.cumsum(y * y)])
    csum_xy = np.concatenate([[0.0], np.cumsum(x * y)])

    n = float(window)
    for j in range(window - 1, m, step):
        i0 = j - window + 1
        i1 = j + 1

        sum_x = csum_x[i1] - csum_x[i0]
        sum_y = csum_y[i1] - csum_y[i0]
        sum_xx = csum_xx[i1] - csum_xx[i0]
        sum_yy = csum_yy[i1] - csum_yy[i0]
        sum_xy = csum_xy[i1] - csum_xy[i0]

        denom = n * sum_xx - sum_x * sum_x
        if denom <= 0:
            continue

        b = (n * sum_xy - sum_x * sum_y) / denom
        if not (0 < b < 1):
            continue

        a = (sum_y - b * sum_x) / n
        sse = (
            sum_yy
            + a * a * n
            + b * b * sum_xx
            + 2 * a * b * sum_x
            - 2 * a * sum_y
            - 2 * b * sum_xy
        )
        if n <= 2:
            continue
        sigma_eps2 = sse / (n - 2)
        if sigma_eps2 <= 0:
            continue

        theta_val = -np.log(b) / dt
        if theta_val <= 0:
            continue

        denom_sigma = 1 - b * b
        if denom_sigma <= 0:
            continue

        sigma_val = np.sqrt(sigma_eps2 * 2 * theta_val / denom_sigma)
        if sigma_val <= 0:
            continue

        mu_val = a / (1 - b)
        idx = j + 1 + lag
        if idx < size:
            mu[idx] = mu_val
            theta[idx] = theta_val
            sigma[idx] = sigma_val

    if forward_fill:
        _forward_fill(mu)
        _forward_fill(theta)
        _forward_fill(sigma)

    return mu, theta, sigma

import torch
import numpy as np



def step(z: tuple[float, float], a: float = 1.4, 
         b: float = 0.3) -> tuple[float, float]:
    """
    Compute the step in the Henon map environment.

    Args:
        z (tuple[float, float]): initial point 
        a (float): controls whether the map exhibits chaotic behavior
        b (float): controls whether the map exhibits chaotic behavior

    Returns:
        tuple[float, float]: next point in trajectory
    """

    x, y = z
    x_next = 1 - a * x ** 2 + y
    y_next = b * x

    return (x_next, y_next)



def rollout(z0: tuple[float, float], K: int) -> list[tuple[float, float]]:
    """
    Recursively computes a trajectory using previous step as an input
    for generating another point.

    Args:
        z0 (tuple[float, float]): Initial point of the trajectory
        K (int): number of steps

    Returns:
        list[tuple[float, float]]: trajectory - a list of 2D points
    """

    traj = [z0]
    z = z0
    for _ in range(K):
        z = step(z)
        traj.append(z)

    return traj



def dist(point_a: tuple[float, float], 
         point_b: tuple[float, float]) -> float:
    """
    Calculates distance between two points.

    Args:
        point_a (tuple[float, float]): 2D point
        point_b (tuple[float, float]): 2D point

    Returns:
        float: Euclidean distance between two points
    """

    x_a, y_a = point_a
    x_b, y_b = point_b
    euc_dist = ((x_a - x_b) ** 2 + (y_a - y_b) ** 2) ** 0.5

    return euc_dist



def compute_drifts(traj_a: list[tuple[float, float]], 
                   traj_b: list[tuple[float, float]]) -> list[float]:
    """
    Compute drifts between two trajectories. Drifts are distances 
    between two points on a same position in compared trajectories.

    Args: 
        traj_a (list[tuple[float, float]]): trajectory of points
        traj_b (list[tuple[float, float]]): trajectory of points

    Returns:
        list[float]: List of drifts
    """
    drifts = []
    for point_a, point_b in zip(traj_a, traj_b):
        distance = dist(point_a, point_b)
        drifts.append(distance)

    return drifts



def is_trajectory_stable(traj: list[tuple[float, float]],
                         threshold: float = 10.0) -> bool:
    """
    Checks if a trajectory is stable. Trajectory is unstable when
    one of it's coordinates is bigger than threshold or is a NaN value.

    Args:
        traj (list[tuple[float, float]]): trajectory, list of 2D points
        threshold (float): number a coordinate must not exceed

    Returns:
        bool: True if trajectory is stable or False if it isn't
    """
    
    return not any(abs(point[0]) > threshold or abs(point[1]) > threshold
                   or np.isnan(point[0]) or np.isnan(point[1]) for point in traj)



def generate_dataset(n_traj: int, len_traj: int, 
                     burn: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a dataset for training model on Henon map. Creates n
    trajectories of a given length, checks if they're stable,
    deletes a given number of first points, because they are not 
    on attractor yet. For every point as an input there is a target point.

    Args:
        n_traj (int): number of trajectories to generate
        len_traj (int): length of each single trajectory
        burn (int): number of points which will be deleted

    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple of inputs and targets
    """

    inputs = []
    targets = []

    for _ in range(0, n_traj):
        z0 = np.random.uniform(-1, 1, 2)
        traj = rollout(z0, len_traj)
        if not is_trajectory_stable(traj):
            continue

        traj = traj[burn:]

        for ipt, target in zip(traj, traj[1:]):
            inputs.append(ipt)
            targets.append(target)

    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)



def normalize_k(k: float, k_min: float = 1, k_max: float = 20) -> float:
    """
    Normalize k number to a range of [-1,1].

    Args:
        k (float): a number to be normalized
        k_min (float): minum value of k number
        k_max (float): maximum value of k number

    Returns:
        float: number normalized to a value between -1 and 1
    """

    return 2 * (k - k_min) / (k_max - k_min) - 1



def generate_horizon_dataset(n_traj: int, len_traj: int,
                             burn: int, k_max: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a dataset for training direct prediction model on 
    Henon map. Creates n trajectories of a given length, checks 
    if they're stable, deletes a given number of first points, because 
    they are not on attractor yet. In the dataset there are many horizons
    for many different starting points.

    Args:
        n_traj (int): number of trajectories to generate
        len_traj (int): length of each single trajectory
        burn (int): number of points which will be deleted
        k_max (int): maximum step

    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple of 
        2D point coordinates and k step
    """

    inputs = []
    targets = []

    for _ in range(0, n_traj):
        z0 = np.random.uniform(-1, 1, 2)
        traj = rollout(z0, len_traj)
        if not is_trajectory_stable(traj):
            continue

        traj = traj[burn:]
        T = len(traj)

        for t in range(T-1):
            steps_till_end = T - 1 - t
            real_k_max = min(k_max, steps_till_end)

            for k in range(1, real_k_max+1):
                k_norm = normalize_k(k=k, k_max=k_max)
                ipt = (*traj[t], k_norm)
                target = traj[t+k]
                inputs.append(ipt)
                targets.append(target)

    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)



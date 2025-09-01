# swarm/validator/task_gen.py
from __future__ import annotations
import random, math
from typing import Tuple, List
from swarm.protocol import MapTask

from swarm.constants import (
    R_MIN, R_MAX, H_MIN, H_MAX,
    MULTI_GOAL_MODE, RANDOM_GOAL_COUNT, 
    MIN_WAYPOINTS, MAX_WAYPOINTS
)
from typing import Optional   

def _goal(seed_rng: random.Random) -> Tuple[float, float, float]:
    ang  = seed_rng.uniform(0, 2*math.pi)
    r    = seed_rng.uniform(R_MIN, R_MAX)
    x, y = r*math.cos(ang), r*math.sin(ang)
    z    = seed_rng.uniform(H_MIN, H_MAX)
    return x, y, z

def _generate_waypoints(
    seed_rng: random.Random, 
    num_waypoints: int,
    min_separation: float = 5.0
) -> List[Tuple[float, float, float]]:
    """Generate waypoints with minimum separation for feasible paths."""
    waypoints = []
    
    for i in range(num_waypoints):
        attempts = 0
        while attempts < 100:
            ang = seed_rng.uniform(0, 2*math.pi)
            r_min_adj = R_MIN + i * 2
            r_max_adj = min(R_MAX, R_MIN + (i + 1) * 8)
            r = seed_rng.uniform(r_min_adj, r_max_adj)
            
            x, y = r*math.cos(ang), r*math.sin(ang)
            z = seed_rng.uniform(H_MIN, H_MAX)
            
            candidate = (x, y, z)
            
            valid = True
            for prev_wp in waypoints:
                dist = math.sqrt(sum((a-b)**2 for a,b in zip(candidate, prev_wp)))
                if dist < min_separation:
                    valid = False
                    break
            
            start_dist = math.sqrt(x*x + y*y + (z-1.5)**2)
            if start_dist < min_separation:
                valid = False
            
            if valid:
                waypoints.append(candidate)
                break
            attempts += 1
        
        if attempts >= 100:
            waypoints.append(candidate)
    
    return waypoints

def random_task(sim_dt: float, horizon: float, seed: Optional[int] = None) -> MapTask:
    if seed is None:
        seed = random.randrange(2**32)
    rng = random.Random(seed)
    
    if MULTI_GOAL_MODE:
        if RANDOM_GOAL_COUNT:
            num_waypoints = rng.randint(MIN_WAYPOINTS, MAX_WAYPOINTS)
        else:
            num_waypoints = MAX_WAYPOINTS
        
        waypoints = _generate_waypoints(rng, num_waypoints)
        base_time_per_wp = 8
        adjusted_horizon = horizon + (num_waypoints - 1) * base_time_per_wp
        
        return MapTask(
            map_seed=seed,
            start=(0.0, 0.0, 1.5),
            goal=waypoints[-1],
            goals=waypoints,
            sim_dt=sim_dt,
            horizon=adjusted_horizon,
            version="1",
        )
    else:
        return MapTask(
            map_seed=seed,
            start=(0.0, 0.0, 1.5),
            goal=_goal(rng),
            sim_dt=sim_dt,
            horizon=horizon,
            version="1",
        )

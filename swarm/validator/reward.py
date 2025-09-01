# swarm/validator/reward.py
"""Reward function for flight missions.

The score is a weighted combination of mission success and time efficiency::

    score = 0.50 * success_term + 0.50 * time_term

where

* ``success_term`` is ``1`` if the mission reaches its goal and ``0``
  otherwise.
* ``time_term`` is based on minimum theoretical time with 2% buffer.

Both weights sum to one. The final score is clamped to ``[0, 1]``.
"""
from __future__ import annotations
import math
import numpy as np
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from swarm.protocol import MapTask

from swarm.constants import SPEED_LIMIT, STABLE_LANDING_SEC

__all__ = ["flight_reward", "multi_waypoint_reward"]


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp *value* to the inclusive range [*lower*, *upper*]."""
    return max(lower, min(upper, value))


def _calculate_target_time(task: "MapTask") -> float:
    """Calculate target time based on distance and 2% buffer."""
    start_pos = np.array(task.start)
    goal_pos = np.array(task.goal)
    distance = np.linalg.norm(goal_pos - start_pos)
    
    min_time = (distance / SPEED_LIMIT) + STABLE_LANDING_SEC
    return min_time * 1.02


def flight_reward(
    success: bool,
    t: float,
    horizon: float,
    task: Optional["MapTask"] = None,
    *,
    w_success: float = 0.5,
    w_t: float = 0.5,
) -> float:
    """Compute the reward for a single flight mission.

    Parameters
    ----------
    success
        ``True`` if the mission successfully reached its objective.
    t
        Time (in seconds) taken to reach the goal.
    horizon
        Maximum time allowed to complete the mission.
    task
        MapTask object containing start and goal positions for distance calculation.
    w_success, w_t
        Weights for the success and time terms. They should sum to ``1``.

    Returns
    -------
    float
        A score in the range ``[0, 1]``.
    """

    if horizon <= 0:
        raise ValueError("'horizon' must be positive")

    success_term = 1.0 if success else 0.0
    
    if success_term == 0.0:
        return 0.0

    if task is not None:
        target_time = _calculate_target_time(task)
        
        if t <= target_time:
            time_term = 1.0
        else:
            time_term = _clamp(1.0 - (t - target_time) / (horizon - target_time))
    else:
        time_term = _clamp(1.0 - t / horizon)

    score = (w_success * success_term) + (w_t * time_term)
    return _clamp(score)


def multi_waypoint_reward(
    waypoints_reached: int,
    total_waypoints: int,
    time_taken: list[float],
    current_time: float,
    horizon: float,
    task: Optional["MapTask"] = None,
) -> float:
    """Compute reward for multi-waypoint navigation missions.

    Parameters
    ----------
    waypoints_reached
        Number of waypoints successfully reached.
    total_waypoints
        Total number of waypoints in the mission.
    time_taken
        List of absolute times when each waypoint was reached.
    current_time
        Current simulation time.
    horizon
        Maximum time allowed for the mission.
    task
        MapTask object containing waypoint positions for distance calculation.

    Returns
    -------
    float
        A score in the range [0, 1].
    """
    if total_waypoints == 0:
        return 0.0

    # Completion ratio - base score for waypoints reached
    completion_ratio = waypoints_reached / total_waypoints

    # Time efficiency calculation
    time_score = 0.0
    if waypoints_reached > 0 and task and hasattr(task, 'goals'):
        # Calculate optimal times for reached segments
        segment_scores = []
        
        for i in range(waypoints_reached):
            # Get segment start and end positions
            if i == 0:
                start_pos = np.array(task.start)
                end_pos = np.array(task.goals[0])
                segment_time = time_taken[0]
            else:
                start_pos = np.array(task.goals[i-1])
                end_pos = np.array(task.goals[i])
                segment_time = time_taken[i] - time_taken[i-1]

            # Calculate optimal time for this segment
            distance = np.linalg.norm(end_pos - start_pos)
            # Hover time: 1s for intermediate waypoints, 3s for final
            hover_time = 3.0 if i == len(task.goals) - 1 else 1.0
            optimal_time = (distance / SPEED_LIMIT) + hover_time
            target_time = optimal_time * 1.02  # 2% buffer

            # Calculate time efficiency for this segment
            if segment_time <= target_time:
                segment_score = 1.0
            else:
                # Remaining time budget for this segment
                remaining = (horizon / total_waypoints) - target_time
                if remaining > 0:
                    segment_score = max(0, 1.0 - (segment_time - target_time) / remaining)
                else:
                    segment_score = 0.5  # Partial credit if no time buffer
            
            segment_scores.append(segment_score)

        time_score = sum(segment_scores) / len(segment_scores) if segment_scores else 0.0
    
    # Calculate final score with different weights based on completion
    if waypoints_reached == total_waypoints:
        # Full completion: success bonus + time efficiency
        score = 0.4 * completion_ratio + 0.5 * time_score + 0.1  # 10% completion bonus
    else:
        # Partial completion: emphasize progress made
        score = 0.7 * completion_ratio + 0.3 * time_score
    
    return _clamp(score)
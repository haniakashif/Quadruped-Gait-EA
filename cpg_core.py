import numpy as np
from typing import Tuple

def update_state_variables(current_val: np.ndarray, target_val: np.ndarray, gamma: float, dt: float) -> np.ndarray:
    derivative = gamma * (target_val - current_val)
    return current_val + (derivative * dt)

def update_global_phases(current_phi_0: np.ndarray, omega: float, w: float, target_offsets: np.ndarray, dt: float) -> np.ndarray:
    n_legs = len(current_phi_0)
    d_phi = np.zeros(n_legs)
    
    for i in range(n_legs):
        coupling_sum = 0
        for j in range(n_legs):
            if i == j: continue
            # eq 12
            phase_diff = target_offsets[j] - target_offsets[i]
            coupling_sum += w * np.sin(current_phi_0[j] - current_phi_0[i] - phase_diff)
        
        d_phi[i] = 2 * np.pi * omega + coupling_sum
        
    return current_phi_0 + (d_phi * dt)

def compute_intra_leg_phases(phi_0: np.ndarray, psi_1: float, psi_2: float) -> Tuple[np.ndarray, np.ndarray]:
    phi_1 = phi_0 + psi_1
    phi_2 = phi_1 + psi_2
    return phi_1, phi_2

def apply_duty_cycle_filter(phi: np.ndarray, d: float) -> np.ndarray:
    phi_2pi = np.mod(phi, 2 * np.pi)
    res = np.zeros_like(phi)
    
    # Stance phase
    mask_stance = phi_2pi < (2 * np.pi * d)
    res[mask_stance] = phi_2pi[mask_stance] / (2 * d)
    
    # Swing phase
    mask_swing = ~mask_stance
    res[mask_swing] = (phi_2pi[mask_swing] + 2 * np.pi * (1 - 2 * d)) / (2 * (1 - d))
    
    return res

def apply_spline_filter(phi_warped: np.ndarray) -> np.ndarray:
    # normalize phase eq 11
    phi_N = 2 * ((phi_warped / (2 * np.pi)) % 0.5)
    res = np.zeros_like(phi_N)
    
    mask = phi_N < 0.5
    
    res[mask] = -16 * (phi_N[mask]**3) + 12 * (phi_N[mask]**2)
    res[~mask] = 16 * ((phi_N[~mask] - 0.5)**3) - 12 * ((phi_N[~mask] - 0.5)**2) + 1
    
    return res

def compute_target_angles(a: np.ndarray, o: np.ndarray, phi_warped: np.ndarray, is_joint_2: bool = False) -> np.ndarray:
    if not is_joint_2:
        # eq 4 for joint 0, 1
        return a * np.cos(phi_warped) + o
    else:
        # eq 8 for joint 2
        return a * phi_warped + o


def clamp_to_joint_limits(angles: np.ndarray) -> np.ndarray:
    
    # Define limits based on the XML structure
    # Hip (Joint 0) range: +/- 0.7853
    # Knee (Joint 1) range: +/- 1.5707
    # Foot (Joint 2) range: +/- 1.5707
    
    lower_limits = np.array([
        -0.7853, -1.5707, -1.5707,  # BL Leg
        -0.7853, -1.5707, -1.5707,  # BR Leg
        -0.7853, -1.5707, -1.5707,  # FL Leg
        -0.7853, -1.5707, -1.5707   # FR Leg
    ])
    
    upper_limits = np.array([
        0.7853, 1.5707, 1.5707,     # BL Leg
        0.7853, 1.5707, 1.5707,     # BR Leg
        0.7853, 1.5707, 1.5707,     # FL Leg
        0.7853, 1.5707, 1.5707      # FR Leg
    ])

    return np.clip(angles, lower_limits, upper_limits)
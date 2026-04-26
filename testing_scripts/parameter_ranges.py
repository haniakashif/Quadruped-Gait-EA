import numpy as np
import matplotlib.pyplot as plt

def F_L(phi, d):
    """
    Duty cycle filter to warp the phase.
    """
    phi_2pi = np.mod(phi, 2 * np.pi)
    
    # Create an array of the same shape to store the results
    result = np.zeros_like(phi_2pi)
    
    # Condition 1: phi_2pi < 2*pi*d
    mask1 = phi_2pi < (2 * np.pi * d)
    result[mask1] = phi_2pi[mask1] / (2 * d)
    
    # Condition 2: otherwise
    mask2 = ~mask1
    result[mask2] = (phi_2pi[mask2] + 2 * np.pi * (1 - 2 * d)) / (2 * (1 - d))
    
    return result

def F_gamma(phi):
    """
    Polynomial filter for Joint 2 to create the double-peak stance/swing curve.
    Uses standard 3rd order splines to ensure velocity continuity.
    """
    # Calculate normalized phase phi_N
    F_L_val = F_L(phi, 0.5) # The filter internally uses a balanced phase for the spline
    phi_N = 2 * ( (F_L_val / (2 * np.pi)) % 0.5 )
    
    result = np.zeros_like(phi_N)
    
    mask1 = phi_N < 0.5
    result[mask1] = -16 * (phi_N[mask1]**3) + 12 * (phi_N[mask1]**2)
    
    mask2 = ~mask1
    result[mask2] = 16 * ((phi_N[mask2] - 1)**3) + 12 * ((phi_N[mask2] - 1)**2)
    
    return result

def main():
    # --- Parameters derived from the paper's Table 1 ---
    omega = 0.25          # Frequency
    d = 0.7               # Duty Cycle [0.2, 0.8] -> 0.7 means long stance, short swing
    
    # Joint 0 Parameters
    a_0 = 0.0             # Target amplitude (static in the paper)
    o_0 = 0.18            # Target offset
    
    # Joint 1 Parameters
    a_1 = 0.3             # Target amplitude [0.0, 0.3]
    o_1 = 0.7             # Target offset [0.36, 1.06]
    psi_1 = 2 * np.pi * 0.05  # Phase shift 2*pi*[-0.1, 0.1]
    
    # Joint 2 Parameters
    a_2_1 = 0.2           # Target swing amplitude [0.0, 0.7]
    a_2_2 = 0.6           # Target stance amplitude [0.0, 0.7]
    o_2 = 0.5             # Target offset [0.85, 1.55]
    psi_2 = 2 * np.pi * -0.05 # Phase shift 2*pi*[-0.1, 0.1]
    
    # --- Time and Phase Simulation ---
    time = np.linspace(0, 10, 500)
    phi_0 = 2 * np.pi * omega * time
    phi_1 = phi_0 + psi_1
    phi_2 = phi_0 + psi_2

    # --- Compute Joint Angles ---
    # Joint 0 & 1
    theta_0 = a_0 * np.cos(F_L(phi_0, d)) + o_0
    theta_1 = a_1 * np.cos(F_L(phi_1, d)) + o_1
    
    # Joint 2 Amplitude Selection
    F_L_phi_2 = F_L(phi_2, d)
    a_2 = np.where(np.mod(F_L_phi_2, 2 * np.pi) < np.pi, a_2_1, a_2_2)
    
    # Joint 2 Angle
    theta_2 = a_2 * F_gamma(phi_2) + o_2

    # --- Plotting ---
    plt.figure(figsize=(12, 8))
    
    plt.plot(time, theta_0, label='Joint 0 (Coxa)', linestyle='--')
    plt.plot(time, theta_1, label='Joint 1 (Femur)', linewidth=2)
    plt.plot(time, theta_2, label='Joint 2 (Tibia)', linewidth=2)
    
    # Add horizontal lines for your robot's physical limits to easily check for clipping
    plt.axhline(y=1.5707, color='r', linestyle=':', label='Max Joint Limit (+1.57 rad)')
    plt.axhline(y=-1.5707, color='r', linestyle=':', label='Min Joint Limit (-1.57 rad)')

    plt.title(f"CPG Joint Trajectories (Duty Cycle = {d})")
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Angle (Radians)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
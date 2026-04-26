"""
This script helps to visualize the end-effector gait of the feet of the robot when set with different joint curves.
"""
    
import numpy as np
import matplotlib.pyplot as plt

# T-Hex Link Lengths (in cm)
L1 = 2.845
L2 = 5.439
L3 = 2.637
L4 = 9.265

def gamma(alpha, p, t, phi, v):
    """
    Periodic joint angle function.
    gamma(a, p, t, phi, v) = a * cos(p*t + phi) + v
    """
    return alpha * np.cos((p * t) + phi) + v

def forward_kinematics(theta1, theta2, theta3, theta4):
    """
    Calculates the 3D position of the foot given the joint angles.
    Supports both scalar and NumPy array inputs.
    """
    sigma_1 = np.sin(theta3 - theta2 + theta4)
    sigma_3 = np.cos(theta3 - theta2 + theta4)
    
    # The intermediate horizontal extension term
    extension = L1 + L4 * sigma_3 + L2 * np.cos(theta2) + L3 * np.cos(theta2 - theta3)
    
    x = np.cos(theta1) * extension
    y = np.sin(theta1) * extension
    z = L2 * np.sin(theta2) - L4 * sigma_1 + L3 * np.sin(theta2 - theta3)
    
    return x, y, z

def main():
    # --- 1. EA Genome Parameters (Now 12 per leg) ---
    
    # Coxa parameters (Controls theta_1 - Lateral Sweep)
    c_alpha = 0.3   # Sweep amplitude (radians)
    c_p = 1         # Period multiplier
    c_phi = 0.0     # Phase alignment
    c_v = 0.0       # Center alignment (0 = pointing straight outward)

    # Femur parameters (Controls theta_2 - Lift/Extension)
    f_alpha = 0.5   
    f_p = 1         
    f_phi = np.pi / 2 # Offset by 90 degrees from the coxa to create a cycle
    f_v = 0.2       

    # Tibia parameters (Controls theta_4 - Tuck/Reach)
    t_alpha = 0.6   
    t_p = 1         
    t_phi = np.pi   # Offset from femur to create the elliptical lift
    t_v = -0.3      

    # Static Joints
    theta3 = np.pi / 4 # Fixed physical geometric offset of the T-Hex knee

    # --- 2. Generate Time Array ---
    # 40 discrete points evaluated over one fundamental period (2*pi)
    t_steps = np.linspace(0, 2 * np.pi, 40)

    # --- 3. Compute Joint Angles ---
    theta1_vals = gamma(c_alpha, c_p, t_steps, c_phi, c_v)
    theta2_vals = gamma(f_alpha, f_p, t_steps, f_phi, f_v)
    theta4_vals = gamma(t_alpha, t_p, t_steps, t_phi, t_v)

    # --- 4. Compute Forward Kinematics ---
    x_vals, y_vals, z_vals = forward_kinematics(theta1_vals, theta2_vals, theta3, theta4_vals)

    # --- 5. Visualization ---
    fig = plt.figure(figsize=(16, 6))

    # Plot 1: The abstract parameter space (Joint Angles)
    ax1 = fig.add_subplot(121)
    ax1.plot(t_steps, theta1_vals, label='Coxa (θ1)', marker='^', color='g')
    ax1.plot(t_steps, theta2_vals, label='Femur (θ2)', marker='o', color='b')
    ax1.plot(t_steps, theta4_vals, label='Tibia (θ4)', marker='s', color='r')
    ax1.set_title("12-Parameter Harmonic Oscillator Output (Genotype)")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("Joint Angle (radians)")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: The physical phenotype (3D Foot Trajectory)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(x_vals, y_vals, z_vals, marker='o', color='g', linewidth=2)
    
    # Mark the start of the stride
    ax2.scatter(x_vals[0], y_vals[0], z_vals[0], color='k', s=100, label='Start (t=0)')
    
    ax2.set_title("Volumetric Foot Trajectory (Phenotype)")
    ax2.set_xlabel("X - Extension (cm)")
    ax2.set_ylabel("Y - Sweep (cm)")
    ax2.set_zlabel("Z - Height (cm)")
    ax2.legend()
    
    # Force equal aspect ratio so the spatial loop isn't distorted
    max_range = np.array([x_vals.max()-x_vals.min(), y_vals.max()-y_vals.min(), z_vals.max()-z_vals.min()]).max() / 2.0
    mid_x = (x_vals.max()+x_vals.min()) * 0.5
    mid_y = (y_vals.max()+y_vals.min()) * 0.5
    mid_z = (z_vals.max()+z_vals.min()) * 0.5
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import cpg_core
import evaluator
import matplotlib.pyplot as plt

os.environ["GDK_BACKEND"] = "x11" 
os.environ["XDG_SESSION_TYPE"] = "x11"
# Create results directory if it doesn't exist
results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)

def run_validation(params: dict, sim_time: float):
    print("Initializing MuJoCo environment...")
    xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
        
    # terrain_data = evaluator.generate_blocky_terrain(nrow=model.hfield_nrow[0], ncol=model.hfield_ncol[0])
    # terrain_data = evaluator.generate_interleaved_terrain(
    #         nrow=model.hfield_nrow[0],
    #         ncol=model.hfield_ncol[0],
    #         verbose=False,
    #     )
    terrain_data = evaluator.generate_randomized_interleaved_terrain(
        nrow=model.hfield_nrow[0],
        ncol=model.hfield_ncol[0],
        verbose=False,
    )
    
    model.hfield_data[:] = terrain_data
    data = mujoco.MjData(model)
    
    mujoco.mj_step(model, data)
    initial_pos = data.body("base_link").xpos.copy()
    body_contact_steps = 0
    total_steps = 0
    
    time_steps = []
    commanded_angles = [[] for _ in range(12)]
    actual_angles = [[] for _ in range(12)]
    actual_torques = [[] for _ in range(12)]
    
    # --- CPG INITIALIZATION ---
    dt = model.opt.timestep
    omega = 0.25 
    
    target_offsets = np.array([0.0, 0.5, 0.25, 0.75]) * 2 * np.pi
    c_phi_0 = target_offsets.copy() 
    c_a0, c_o0 = np.zeros(4), np.zeros(4)
    c_a1, c_o1 = np.zeros(4), np.zeros(4)
    c_a2_1, c_a2_2, c_o2 = np.zeros(4), np.zeros(4), np.zeros(4)
    
    t_a0 = np.full(4, params['mu_r0'])
    t_o0 = np.full(4, params['mu_o0'])
    t_a1 = np.full(4, params['mu_r1'])
    t_o1 = np.full(4, params['mu_o1'])
    t_a2_1 = np.full(4, params['mu_r2_1'])
    t_a2_2 = np.full(4, params['mu_r2_2'])
    t_o2 = np.full(4, params['mu_o2'])
    
    print("Launching Viewer. Close the window to terminate early.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time <= sim_time:
            step_start = time.time()
            
            c_a0 = cpg_core.update_state_variables(c_a0, t_a0, params['gamma'], dt)
            c_o0 = cpg_core.update_state_variables(c_o0, t_o0, params['gamma'], dt)
            c_a1 = cpg_core.update_state_variables(c_a1, t_a1, params['gamma'], dt)
            c_o1 = cpg_core.update_state_variables(c_o1, t_o1, params['gamma'], dt)
            c_a2_1 = cpg_core.update_state_variables(c_a2_1, t_a2_1, params['gamma'], dt)
            c_a2_2 = cpg_core.update_state_variables(c_a2_2, t_a2_2, params['gamma'], dt)
            c_o2 = cpg_core.update_state_variables(c_o2, t_o2, params['gamma'], dt)

            c_phi_0 = cpg_core.update_global_phases(c_phi_0, omega, params['coupling_w'], target_offsets, dt)
            phi_1, phi_2 = cpg_core.compute_intra_leg_phases(c_phi_0, params['psi_1'], params['psi_2'])

            phi_0_w = cpg_core.apply_duty_cycle_filter(c_phi_0, params['duty_cycle'])
            phi_1_w = cpg_core.apply_duty_cycle_filter(phi_1, params['duty_cycle'])
            phi_2_w = cpg_core.apply_duty_cycle_filter(phi_2, params['duty_cycle'])

            phi_2_2pi = np.mod(phi_2_w, 2 * np.pi)
            c_a2 = np.where(phi_2_2pi < np.pi, c_a2_1, c_a2_2) 
            phi_2_spline = cpg_core.apply_spline_filter(phi_2_w)

            theta_0 = cpg_core.compute_target_angles(c_a0, c_o0, phi_0_w, False)
            theta_1 = cpg_core.compute_target_angles(c_a1, c_o1, phi_1_w, False)
            theta_2 = cpg_core.compute_target_angles(c_a2, c_o2, phi_2_spline, True)

            raw_angles = np.zeros(12)
            raw_angles[0:3]  = [-theta_0[0], -theta_1[0], -theta_2[0]] # BL
            raw_angles[3:6]  = [ theta_0[1],  theta_1[1],  theta_2[1]] # BR
            raw_angles[6:9]  = [-theta_0[2], -theta_1[2], -theta_2[2]] # FL
            raw_angles[9:12] = [ theta_0[3],  theta_1[3],  theta_2[3]] # FR
            
            data.ctrl[:] = cpg_core.clamp_to_joint_limits(raw_angles)
            mujoco.mj_step(model, data)
            
            time_steps.append(data.time)
            for i in range(12):
                commanded_angles[i].append(raw_angles[i])  
                actual_angles[i].append(data.qpos[i + 7])  
                actual_torques[i].append(data.qfrc_actuator[i + 6])
            
            total_steps += 1
            if evaluator.has_forbidden_terrain_contact(model, data):
                body_contact_steps += 1
            
            current_y = float(data.body("base_link").xpos[1])
            if current_y >= 7.2:  # if robot has reached near end of terrain
                print(f"Reached end of terrain at time {data.time:.2f}s, Y={current_y:.4f}m. Ending simulation.")
                break
            
            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        viewer.close() 

    final_dx = data.body("base_link").xpos[0] - initial_pos[0]
    final_dy = data.body("base_link").xpos[1] - initial_pos[1]
    body_contact_fraction = (
        body_contact_steps / total_steps if total_steps > 0 else 0.0
    )
    
    drift_penalty_weight = 2.0 
    fitness = final_dy - (drift_penalty_weight * abs(final_dx))
    
    print("\n" + "="*40)
    print(f"Forward Travel (Y): {final_dy:.4f} m")
    print(f"Lateral Drift (X):  {final_dx:.4f} m")
    print(f"Forbidden Contacts: {body_contact_fraction:.2%}")
    print(f"Final Fitness:      {fitness:.4f}")
    print("="*40 + "\n")
    
    leg_names = ["BL", "BL", "BL", "BR", "BR", "BR", "FL", "FL", "FL", "FR", "FR", "FR"]
    joint_names = ["Hip", "Knee", "Ankle"]
    
    # Plot Joint Angles
    fig_angles, axes_angles = plt.subplots(4, 3, figsize=(16, 12))
    fig_angles.suptitle("Joint Angles", fontsize=16, fontweight='bold')
    
    for i in range(12):
        ax = axes_angles[i // 3, i % 3]
        ax.plot(time_steps, commanded_angles[i], label="Commanded Angles", linewidth=1.5, color='blue')
        ax.set_title(f"{leg_names[i]} - {joint_names[i % 3]}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (rad)")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "joint_angles.png"), dpi=150, bbox_inches='tight')
    print("Saved joint_angles.png")
    
    fig_actual_angles, axes_actual_angles = plt.subplots(4, 3, figsize=(16, 12))
    fig_actual_angles.suptitle("Actual Joint Angles", fontsize=16, fontweight='bold')
    
    for i in range(12):
        ax = axes_actual_angles[i // 3, i % 3]
        ax.plot(time_steps, actual_angles[i], label="Actual Angles", linewidth=1.5, color='green')
        ax.set_title(f"{leg_names[i]} - {joint_names[i % 3]}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (rad)")
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "actual_joint_angles.png"), dpi=150, bbox_inches='tight')
    print("Saved actual_joint_angles.png")
    
    # Plot Motor Torques
    fig_torques, axes_torques = plt.subplots(4, 3, figsize=(16, 12))
    fig_torques.suptitle("Servo Torques", fontsize=16, fontweight='bold')
    
    for i in range(12):
        ax = axes_torques[i // 3, i % 3]
        ax.plot(time_steps, actual_torques[i], label="Actual Torque", linewidth=1.5, color='red')
        ax.set_title(f"{leg_names[i]} - {joint_names[i % 3]}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Torque (N·m)")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "servo_torques.png"), dpi=150, bbox_inches='tight')
    print("Saved servo_torques.png")
    
    # --- ALTERNATIVE TORQUE PLOTTING: Per-Leg Detailed View ---
    # Comment out the previous torque section to use this instead
    
    leg_names_full = ["BL", "BR", "FL", "FR"]
    joint_indices = {
        "BL": [0, 1, 2],
        "BR": [3, 4, 5],
        "FL": [6, 7, 8],
        "FR": [9, 10, 11]
    }
    joint_names = ["Hip", "Knee", "Ankle"]
    
    for leg_idx, leg_name in enumerate(leg_names_full):
        fig_leg, axes_leg = plt.subplots(3, 1, figsize=(18, 10))
        fig_leg.suptitle(f"{leg_name} Leg - Joint Torques", fontsize=16, fontweight='bold')
        
        joint_indices_for_leg = joint_indices[leg_name]
        
        for joint_pos, joint_idx in enumerate(joint_indices_for_leg):
            ax = axes_leg[joint_pos]
            ax.plot(time_steps, actual_torques[joint_idx], linewidth=1.2, color='red', alpha=0.8)
            ax.set_title(f"{leg_name} - {joint_names[joint_pos]} Joint", fontsize=12, fontweight='bold')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Torque (N·m)")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"torque_{leg_name}.png"), dpi=150, bbox_inches='tight')
        print(f"Saved torque_{leg_name}.png")
    
    plt.close('all')

if __name__ == "__main__":
    optimized_params = {
        "gamma": 0.33242948003377987,
        "duty_cycle": 0.4172964738872599,
        "coupling_w": 1.814359554786942,
        "mu_r0": 0.5977562499535333,
        "mu_o0": -0.06821011367710861,
        "psi_1": 0.6283185307179586,
        "mu_r1": 0.2878435438298354,
        "mu_o1": 0.4279269256411492,
        "psi_2": -0.47850751097630134,
        "mu_r2_1": 0.697308337055854,
        "mu_r2_2": 0.007488975240002193,
        "mu_o2": 0.85
    }
    
    run_validation(optimized_params, sim_time=150.0)

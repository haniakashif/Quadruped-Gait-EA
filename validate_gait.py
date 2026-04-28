import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import cpg_core

# --- LINUX WAYLAND FIX ---
# Forces GLFW to use X11 backend to prevent window destruction deadlocks
os.environ["GDK_BACKEND"] = "x11" 
os.environ["XDG_SESSION_TYPE"] = "x11"

def generate_blocky_terrain(nrow, ncol):
    print("Generating Terrain ...")
    terrain = np.zeros((nrow, ncol))
    
    zone1_end = nrow // 3      
    zone2_end = 2 * nrow // 3  
    
    mod_block = 1
    mod_max_height = 0.5  
    for i in range(zone1_end, zone2_end, mod_block):
        for j in range(0, ncol, mod_block):
            terrain[i:i+mod_block, j:j+mod_block] = np.random.uniform(0.0, mod_max_height)

    hard_block = 1
    hard_max_height = 1.0 
    for i in range(zone2_end, nrow, hard_block):
        for j in range(0, ncol, hard_block):
            terrain[i:i+hard_block, j:j+hard_block] = np.random.uniform(0.0, hard_max_height)
            
    return terrain.flatten()

def run_validation(params: dict, sim_time: float):
    """
    Executes a MuJoCo simulation using a deterministic set of CPG parameters.
    """
    print("Initializing MuJoCo environment...")
    xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
        
    terrain_data = generate_blocky_terrain(nrow=model.hfield_nrow[0], ncol=model.hfield_ncol[0])
    model.hfield_data[:] = terrain_data
    data = mujoco.MjData(model)
    
    mujoco.mj_step(model, data)
    initial_pos = data.body("base_link").xpos.copy()
    
    # --- CPG INITIALIZATION ---
    dt = model.opt.timestep
    omega = 0.25 
    
    # 1. Target Walking Gait Offsets
    target_offsets = np.array([0.0, 0.5, 0.25, 0.75]) * 2 * np.pi
    
    # 2. Bootstrap the master clocks to avoid transient swaying
    c_phi_0 = target_offsets.copy() 
    
    # 3. Initialize current state arrays at zero for smooth ODE ramp-up
    c_a0, c_o0 = np.zeros(4), np.zeros(4)
    c_a1, c_o1 = np.zeros(4), np.zeros(4)
    c_a2_1, c_a2_2, c_o2 = np.zeros(4), np.zeros(4), np.zeros(4)
    
    # 4. Map the provided physical parameters to the target arrays
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
            
            # --- CPG DYNAMICAL SYSTEM ---
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

            # --- KINEMATIC INVERSION FOR LEFT LEGS ---
            raw_angles = np.zeros(12)
            raw_angles[0:3]  = [-theta_0[0], -theta_1[0], -theta_2[0]] # BL
            raw_angles[3:6]  = [ theta_0[1],  theta_1[1],  theta_2[1]] # BR
            raw_angles[6:9]  = [-theta_0[2], -theta_1[2], -theta_2[2]] # FL
            raw_angles[9:12] = [ theta_0[3],  theta_1[3],  theta_2[3]] # FR
            
            data.ctrl[:] = cpg_core.clamp_to_joint_limits(raw_angles)
            
            # Step physics and render
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Clock synchronization
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        viewer.close() 

    # --- PERFORMANCE DIAGNOSTICS ---
    final_dx = data.body("base_link").xpos[0] - initial_pos[0]
    final_dy = data.body("base_link").xpos[1] - initial_pos[1]
    
    drift_penalty_weight = 2.0 
    fitness = final_dy - (drift_penalty_weight * abs(final_dx))
    
    print("\n" + "="*40)
    print(" VALIDATION COMPLETE ")
    print("="*40)
    print(f"Forward Travel (Y): {final_dy:.4f} m")
    print(f"Lateral Drift (X):  {final_dx:.4f} m")
    print(f"Final Fitness:      {fitness:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    # Your optimized parameters extracted from the JSON output
    optimized_params = {
        "gamma": 0.4366072853760319,
        "duty_cycle": 0.6831063422956241,
        "coupling_w": 1.2382489322537358,
        "mu_r0": 0.5744139650554527,
        "mu_o0": -0.20508986741261034,
        "psi_1": 0.23755152260925955,
        "mu_r1": 0.2471750831292964,
        "mu_o1": 0.6361297713864255,
        "psi_2": 0.12393590935342069,
        "mu_r2_1": 0.7,
        "mu_r2_2": 0.0,
        "mu_o2": 0.8502197101284521
    }
    
    run_validation(optimized_params, sim_time=200.0)
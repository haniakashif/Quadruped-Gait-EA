import os
import multiprocessing as mp
import numpy as np
import mujoco
import time
import mujoco.viewer
import cpg_core

# --- LINUX WAYLAND FIX ---
# Forces GLFW to use X11 backend to prevent window destruction deadlocks
os.environ["GDK_BACKEND"] = "x11" 
os.environ["XDG_SESSION_TYPE"] = "x11"

def decode_genome(genome: np.ndarray) -> dict:
    return {
        "gamma":           0.2 + genome[0] * (0.6 - 0.2),
        "duty_cycle":      0.2 + genome[1] * (0.8 - 0.2),
        "coupling_w":      0.1 + genome[2] * (2.0 - 0.1),
        
        "mu_r0":           0.0 + genome[3] * (0.6 - 0.0), 
        "mu_o0":          -0.4 + genome[4] * (0.4 - (-0.4)),
        
        "psi_1":           2 * np.pi * (-0.1 + genome[5] * (0.1 - (-0.1))),
        "mu_r1":           0.0 + genome[6] * (0.3 - 0.0),
        "mu_o1":           0.36 + genome[7] * (1.06 - 0.36),
        
        "psi_2":           2 * np.pi * (-0.1 + genome[8] * (0.1 - (-0.1))),
        "mu_r2_1":         0.0 + genome[9] * (0.7 - 0.0),
        "mu_r2_2":         0.0 + genome[10] * (0.7 - 0.0),
        "mu_o2":           0.85 + genome[11] * (1.55 - 0.85)
    }

# def generate_blocky_terrain(nrow, ncol):
#     """Generates the 3-Zone Y-Oriented Testbed"""
#     terrain = np.zeros((nrow, ncol))
#     zone1_end = nrow // 3
#     zone2_end = 2 * nrow // 3
    
#     for i in range(zone1_end, zone2_end):
#         for j in range(0, ncol):
#             terrain[i, j] = np.random.uniform(0.0, 0.5)

#     for i in range(zone2_end, nrow):
#         for j in range(0, ncol):
#             terrain[i, j] = np.random.uniform(0.0, 1.0)
            
#     return terrain.flatten()
def generate_blocky_terrain(nrow, ncol):
    print(f"Generating Terrain ...")
    # np.random.seed(seed)  # Lock the random generation to match the headless ID
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


def simulate_universe(args: tuple) -> float:
    """The isolated worker thread running the CPG and physics loop."""
    robot_id, lock, genome = args
    
    try:
        params = decode_genome(genome)
        
        xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
        with lock:
            model = mujoco.MjModel.from_xml_path(xml_path)
            
        terrain_data = generate_blocky_terrain(nrow=model.hfield_nrow[0], ncol=model.hfield_ncol[0])
        model.hfield_data[:] = terrain_data
        data = mujoco.MjData(model)
        
        mujoco.mj_step(model, data)
        initial_pos = data.body("base_link").xpos.copy()
        
        
        # --- CPG INITIALIZATION ---
        dt = model.opt.timestep
        omega = 0.25 # Fixed frequency
        
        # State variables for 4 legs: [BL, BR, FL, FR]
        c_phi_0 = np.zeros(4)
        c_a0, c_o0 = np.zeros(4), np.zeros(4)
        c_a1, c_o1 = np.zeros(4), np.zeros(4)
        c_a2_1, c_a2_2, c_o2 = np.zeros(4), np.zeros(4), np.zeros(4)
        
        # Target arrays mapped across all 4 legs to enforce symmetry
        t_a0 = np.full(4, params['mu_r0'])
        t_o0 = np.full(4, params['mu_o0'])
        t_a1 = np.full(4, params['mu_r1'])
        t_o1 = np.full(4, params['mu_o1'])
        t_a2_1 = np.full(4, params['mu_r2_1'])
        t_a2_2 = np.full(4, params['mu_r2_2'])
        t_o2 = np.full(4, params['mu_o2'])
        
        # Static phase offsets for a Walk Gait (Eq 12)
        target_offsets = np.array([0.0, 0.5, 0.25, 0.75]) * 2 * np.pi
        
        max_sim_time = 20.0
        
        # --- MAIN PHYSICS LOOP ---
        while data.time <= max_sim_time:
            # 1. Update internal state variables (Smooth ODEs)
            c_a0 = cpg_core.update_state_variables(c_a0, t_a0, params['gamma'], dt)
            c_o0 = cpg_core.update_state_variables(c_o0, t_o0, params['gamma'], dt)
            c_a1 = cpg_core.update_state_variables(c_a1, t_a1, params['gamma'], dt)
            c_o1 = cpg_core.update_state_variables(c_o1, t_o1, params['gamma'], dt)
            c_a2_1 = cpg_core.update_state_variables(c_a2_1, t_a2_1, params['gamma'], dt)
            c_a2_2 = cpg_core.update_state_variables(c_a2_2, t_a2_2, params['gamma'], dt)
            c_o2 = cpg_core.update_state_variables(c_o2, t_o2, params['gamma'], dt)

            # 2. Update timing
            c_phi_0 = cpg_core.update_global_phases(c_phi_0, omega, params['coupling_w'], target_offsets, dt)
            phi_1, phi_2 = cpg_core.compute_intra_leg_phases(c_phi_0, params['psi_1'], params['psi_2'])

            # 3. Apply Time Warp
            phi_0_w = cpg_core.apply_duty_cycle_filter(c_phi_0, params['duty_cycle'])
            phi_1_w = cpg_core.apply_duty_cycle_filter(phi_1, params['duty_cycle'])
            phi_2_w = cpg_core.apply_duty_cycle_filter(phi_2, params['duty_cycle'])

            # 4. Process Joint 2 Spline & Amplitude Selection
            phi_2_2pi = np.mod(phi_2_w, 2 * np.pi)
            c_a2 = np.where(phi_2_2pi < np.pi, c_a2_1, c_a2_2) # Switch between swing and stance amplitude
            phi_2_spline = cpg_core.apply_spline_filter(phi_2_w)

            # 5. Generate target angles
            theta_0 = cpg_core.compute_target_angles(c_a0, c_o0, phi_0_w, False)
            theta_1 = cpg_core.compute_target_angles(c_a1, c_o1, phi_1_w, False)
            theta_2 = cpg_core.compute_target_angles(c_a2, c_o2, phi_2_spline, True)

            # 6. Interleave angles to match XML actuator order and clamp
            raw_angles = np.zeros(12)
            raw_angles[0:3] = [theta_0[0], theta_1[0], theta_2[0]] # BL
            raw_angles[3:6] = [theta_0[1], theta_1[1], theta_2[1]] # BR
            raw_angles[6:9] = [theta_0[2], theta_1[2], theta_2[2]] # FL
            raw_angles[9:12] = [theta_0[3], theta_1[3], theta_2[3]] # FR
            
            data.ctrl[:] = cpg_core.clamp_to_joint_limits(raw_angles)

            # Advance physics
            mujoco.mj_step(model, data)
                
        # --- FITNESS CALCULATION ---
        final_dx = data.body("base_link").xpos[0] - initial_pos[0]
        final_dy = data.body("base_link").xpos[1] - initial_pos[1]
        
        drift_penalty_weight = 2.0 
        fitness = final_dy - (drift_penalty_weight * abs(final_dx))
        
        return float(fitness)
        
    except Exception as e:
        print(f"FAILED: Universe {robot_id:02d} threw an error: {e}")
        # Return a heavily penalized score so unstable genomes are eliminated
        return -999.0


def run_headless_pool(population: np.ndarray, max_workers: int = None) -> np.ndarray:
    if max_workers is None:
        max_workers = os.cpu_count()
        
    manager = mp.Manager()
    lock = manager.Lock()
    
    tasks = [(i, lock, population[i]) for i in range(len(population))]
    
    with mp.Pool(processes=max_workers) as pool:
        fitness_scores = pool.map(simulate_universe, tasks)
        
    return np.array(fitness_scores)





# def visualize_genome(genome: np.ndarray, sim_time: float = 20.0):
#     """
#     Renders a single genome in a MuJoCo GUI window.
#     Uses the exact same CPG mathematics as the headless evaluator.
#     """
#     print("Launching MuJoCo Viewer... (Close the window to continue)")
    
#     # 1. Decode just like the headless version
#     params = decode_genome(genome)
    
#     xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
#     model = mujoco.MjModel.from_xml_path(xml_path)
        
#     terrain_data = generate_blocky_terrain(nrow=model.hfield_nrow[0], ncol=model.hfield_ncol[0])
#     model.hfield_data[:] = terrain_data
#     data = mujoco.MjData(model)
    
#     mujoco.mj_step(model, data)
#     initial_pos = data.body("base_link").xpos.copy()
    
#     # 2. Exact CPG Initialization
#     dt = model.opt.timestep
#     omega = 0.25 
    
#     c_phi_0 = np.zeros(4)
#     c_a0, c_o0 = np.zeros(4), np.zeros(4)
#     c_a1, c_o1 = np.zeros(4), np.zeros(4)
#     c_a2_1, c_a2_2, c_o2 = np.zeros(4), np.zeros(4), np.zeros(4)
    
#     t_a0 = np.full(4, params['mu_r0'])
#     t_o0 = np.full(4, params['mu_o0'])
#     t_a1 = np.full(4, params['mu_r1'])
#     t_o1 = np.full(4, params['mu_o1'])
#     t_a2_1 = np.full(4, params['mu_r2_1'])
#     t_a2_2 = np.full(4, params['mu_r2_2'])
#     t_o2 = np.full(4, params['mu_o2'])
    
#     target_offsets = np.array([0.0, 0.5, 0.25, 0.75]) * 2 * np.pi
    
#     # 3. Launch the Passive Viewer
#     with mujoco.viewer.launch_passive(model, data) as viewer:
#         while viewer.is_running() and data.time <= sim_time:
#             step_start = time.time()
            
#             # --- Exact CPG Math ---
#             c_a0 = cpg_core.update_state_variables(c_a0, t_a0, params['gamma'], dt)
#             c_o0 = cpg_core.update_state_variables(c_o0, t_o0, params['gamma'], dt)
#             c_a1 = cpg_core.update_state_variables(c_a1, t_a1, params['gamma'], dt)
#             c_o1 = cpg_core.update_state_variables(c_o1, t_o1, params['gamma'], dt)
#             c_a2_1 = cpg_core.update_state_variables(c_a2_1, t_a2_1, params['gamma'], dt)
#             c_a2_2 = cpg_core.update_state_variables(c_a2_2, t_a2_2, params['gamma'], dt)
#             c_o2 = cpg_core.update_state_variables(c_o2, t_o2, params['gamma'], dt)

#             c_phi_0 = cpg_core.update_global_phases(c_phi_0, omega, params['coupling_w'], target_offsets, dt)
#             phi_1, phi_2 = cpg_core.compute_intra_leg_phases(c_phi_0, params['psi_1'], params['psi_2'])

#             phi_0_w = cpg_core.apply_duty_cycle_filter(c_phi_0, params['duty_cycle'])
#             phi_1_w = cpg_core.apply_duty_cycle_filter(phi_1, params['duty_cycle'])
#             phi_2_w = cpg_core.apply_duty_cycle_filter(phi_2, params['duty_cycle'])

#             phi_2_2pi = np.mod(phi_2_w, 2 * np.pi)
#             c_a2 = np.where(phi_2_2pi < np.pi, c_a2_1, c_a2_2) 
#             phi_2_spline = cpg_core.apply_spline_filter(phi_2_w)

#             theta_0 = cpg_core.compute_target_angles(c_a0, c_o0, phi_0_w, False)
#             theta_1 = cpg_core.compute_target_angles(c_a1, c_o1, phi_1_w, False)
#             theta_2 = cpg_core.compute_target_angles(c_a2, c_o2, phi_2_spline, True)

#             raw_angles = np.zeros(12)
#             raw_angles[0:3] = [theta_0[0], theta_1[0], theta_2[0]] 
#             raw_angles[3:6] = [theta_0[1], theta_1[1], theta_2[1]] 
#             raw_angles[6:9] = [theta_0[2], theta_1[2], theta_2[2]] 
#             raw_angles[9:12] = [theta_0[3], theta_1[3], theta_2[3]] 
            
#             data.ctrl[:] = cpg_core.clamp_to_joint_limits(raw_angles)
            
#             # Advance physics and update the GUI
#             mujoco.mj_step(model, data)
#             viewer.sync()
            
#             # Sync with real-time so the simulation doesn't finish in 0.1 seconds
#             time_until_next_step = dt - (time.time() - step_start)
#             if time_until_next_step > 0:
#                 time.sleep(time_until_next_step)
                
                
#     # --- DIAGNOSTIC FITNESS EVALUATION ---
#     # This matches simulate_universe exactly, but prints instead of returns
#     final_dx = data.body("base_link").xpos[0] - initial_pos[0]
#     final_dy = data.body("base_link").xpos[1] - initial_pos[1]
    
#     drift_penalty_weight = 2.0 
#     fitness = final_dy - (drift_penalty_weight * abs(final_dx))
    
#     print("-" * 40)
#     print("VISUALIZATION COMPLETE")
#     print(f"Forward Travel (Y): {final_dy:.4f} m")
#     print(f"Lateral Drift (X):  {final_dx:.4f} m")
#     print(f"Calculated Fitness: {fitness:.4f}")
#     print("-" * 40)




def visualize_genome(genome: np.ndarray, sim_time: float = 20.0, robot_id: int = 0) -> float:
    """
    Renders a single genome in a MuJoCo GUI window, calculates fitness, 
    safely closes the window, and returns the score to the EA.
    """
    params = decode_genome(genome)
    
    xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
        
    terrain_data = generate_blocky_terrain(nrow=model.hfield_nrow[0], ncol=model.hfield_ncol[0])
    model.hfield_data[:] = terrain_data
    data = mujoco.MjData(model)
    
    mujoco.mj_step(model, data)
    initial_pos = data.body("base_link").xpos.copy()
    
    # CPG Initialization
    dt = model.opt.timestep
    omega = 0.25 
    
    c_phi_0 = np.zeros(4)
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
    
    target_offsets = np.array([0.0, 0.5, 0.25, 0.75]) * 2 * np.pi
    
    # Launch the Passive Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time <= sim_time:
            step_start = time.time()
            
            # --- Exact CPG Math ---
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
            raw_angles[0:3] = [theta_0[0], theta_1[0], theta_2[0]] 
            raw_angles[3:6] = [theta_0[1], theta_1[1], theta_2[1]] 
            raw_angles[6:9] = [theta_0[2], theta_1[2], theta_2[2]] 
            raw_angles[9:12] = [theta_0[3], theta_1[3], theta_2[3]] 
            
            data.ctrl[:] = cpg_core.clamp_to_joint_limits(raw_angles)
            
            # Advance physics
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Sync to real-time clock
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        # Explicitly kill the viewer thread to bypass Linux Wayland deadlocks
        viewer.close() 

    # --- FITNESS EVALUATION ---
    final_dx = data.body("base_link").xpos[0] - initial_pos[0]
    final_dy = data.body("base_link").xpos[1] - initial_pos[1]
    
    drift_penalty_weight = 2.0 
    fitness = final_dy - (drift_penalty_weight * abs(final_dx))
    
    print(f"    [Robot {robot_id:02d}] Finished. Travel(Y): {final_dy:.3f}m | Drift(X): {final_dx:.3f}m | Fitness: {fitness:.3f}")
    
    return float(fitness)


def run_visual_sequential(population: np.ndarray) -> np.ndarray:
    """
    Evaluates the population one by one in a GUI window instead of headless parallel threads.
    """
    fitness_scores = []
    for i, genome in enumerate(population):
        print(f"  -> Launching Viewer for Robot {i}/{len(population)} ...")
        fit = visualize_genome(genome, sim_time=20.0, robot_id=i)
        fitness_scores.append(fit)
        
    return np.array(fitness_scores)
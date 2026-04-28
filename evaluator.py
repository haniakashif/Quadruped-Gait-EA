import os
import multiprocessing as mp
import numpy as np
import mujoco
import time
import mujoco.viewer
import cpg_core

# Forces GLFW to use X11 backend to prevent window destruction deadlocks
os.environ["GDK_BACKEND"] = "x11" 
os.environ["XDG_SESSION_TYPE"] = "x11"

HEIGHT_PENALTY_WEIGHT = 1.5
SETTLE_STEPS = 100
_FLAT_GROUND_BASE_HEIGHT = None


def get_flat_ground_base_height(xml_path: str, settle_steps: int = SETTLE_STEPS) -> float:
    global _FLAT_GROUND_BASE_HEIGHT

    if _FLAT_GROUND_BASE_HEIGHT is not None:
        return _FLAT_GROUND_BASE_HEIGHT

    model = mujoco.MjModel.from_xml_path(xml_path)
    model.hfield_data[:] = 0.0

    data = mujoco.MjData(model)
    data.ctrl[:] = 0.0

    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    _FLAT_GROUND_BASE_HEIGHT = float(data.body("base_link").xpos[2])
    return _FLAT_GROUND_BASE_HEIGHT

def decode_genome(genome: np.ndarray) -> dict:
    return {
        # all params as per paper
        "gamma":           0.2 + genome[0] * (0.6 - 0.2),
        "duty_cycle":      0.2 + genome[1] * (0.8 - 0.2),
        "coupling_w":      0.1 + genome[2] * (2.0 - 0.1),
        
        # varying mu_r0 & mu_o0 ourselves as its not part of authors' approach
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


def format_result_log(robot_id: int, dy: float, dx: float, height_dev: float, fitness: float, failed: bool = False, error: str = "") -> str:
    status = "FAILED" if failed else "Finished"
    message = (
        f"    [Robot {robot_id:02d}] {status}. "
        f"Travel(Y): {dy:.3f}m | Drift(X): {dx:.3f}m | "
        f"Height Deviation (Z): {height_dev:.3f}m | Fitness: {fitness:.3f}"
    )
    if error:
        message += f" | Error: {error}"
    return message


def generate_blocky_terrain(nrow, ncol, verbose=True):
    if verbose:
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


def simulate_universe(args: tuple):
    robot_id, lock, genome = args
    
    try:
        params = decode_genome(genome)
        
        xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
        flat_ground_height = get_flat_ground_base_height(xml_path)

        with lock:
            model = mujoco.MjModel.from_xml_path(xml_path)
            
        terrain_data = generate_blocky_terrain(
            nrow=model.hfield_nrow[0],
            ncol=model.hfield_ncol[0],
            verbose=False,
        )
        model.hfield_data[:] = terrain_data
        data = mujoco.MjData(model)
        
        mujoco.mj_step(model, data)
        initial_pos = data.body("base_link").xpos.copy()
        height_deviation_sum = 0.0
        height_samples = 0
        current_height = float(data.body("base_link").xpos[2])
        height_deviation_sum += abs(current_height - flat_ground_height)
        height_samples += 1
        
        # CPG initialization
        dt = model.opt.timestep # integration timestep from XML
        omega = 0.25 # constant in the paper
        
        # actual amplitude, offset and phase variables
        target_offsets = np.array([0.0, 0.5, 0.25, 0.75]) * 2 * np.pi
        c_phi_0 = target_offsets.copy()
        
        c_a0, c_o0 = np.zeros(4), np.zeros(4)
        c_a1, c_o1 = np.zeros(4), np.zeros(4)
        c_a2_1, c_a2_2, c_o2 = np.zeros(4), np.zeros(4), np.zeros(4)
        
        # target arrays mapped across all legs for symmetry
        t_a0 = np.full(4, params['mu_r0'])
        t_o0 = np.full(4, params['mu_o0'])
        t_a1 = np.full(4, params['mu_r1'])
        t_o1 = np.full(4, params['mu_o1'])
        t_a2_1 = np.full(4, params['mu_r2_1'])
        t_a2_2 = np.full(4, params['mu_r2_2'])
        t_o2 = np.full(4, params['mu_o2'])
        
        # static phase offsets for leg
        target_offsets = np.array([0.0, 0.5, 0.25, 0.75]) * 2 * np.pi
        sim_time = 200.0
        
        while data.time <= sim_time:
            # derivative updates for smooth stage change (eq 1, 2, 5, 6, 7)
            c_a0 = cpg_core.update_state_variables(c_a0, t_a0, params['gamma'], dt)
            c_o0 = cpg_core.update_state_variables(c_o0, t_o0, params['gamma'], dt)
            c_a1 = cpg_core.update_state_variables(c_a1, t_a1, params['gamma'], dt)
            c_o1 = cpg_core.update_state_variables(c_o1, t_o1, params['gamma'], dt)
            c_a2_1 = cpg_core.update_state_variables(c_a2_1, t_a2_1, params['gamma'], dt)
            c_a2_2 = cpg_core.update_state_variables(c_a2_2, t_a2_2, params['gamma'], dt)
            c_o2 = cpg_core.update_state_variables(c_o2, t_o2, params['gamma'], dt)

            # synchonize hips of all legs (eq 12)
            c_phi_0 = cpg_core.update_global_phases(c_phi_0, omega, params['coupling_w'], target_offsets, dt)
            # phase diff bw joints in the same leg
            phi_1, phi_2 = cpg_core.compute_intra_leg_phases(c_phi_0, params['psi_1'], params['psi_2'])

            # F_l filter
            phi_0_w = cpg_core.apply_duty_cycle_filter(c_phi_0, params['duty_cycle'])
            phi_1_w = cpg_core.apply_duty_cycle_filter(phi_1, params['duty_cycle'])
            phi_2_w = cpg_core.apply_duty_cycle_filter(phi_2, params['duty_cycle'])

            # amplitude selection for joint 2 and spline filter
            phi_2_2pi = np.mod(phi_2_w, 2 * np.pi) # selecting amplitude based on F_l filter from prev
            c_a2 = np.where(phi_2_2pi < np.pi, c_a2_1, c_a2_2) # switch between swing and stance amplitude
            phi_2_spline = cpg_core.apply_spline_filter(phi_2_w)

            theta_0 = cpg_core.compute_target_angles(c_a0, c_o0, phi_0_w, False)
            theta_1 = cpg_core.compute_target_angles(c_a1, c_o1, phi_1_w, False)
            theta_2 = cpg_core.compute_target_angles(c_a2, c_o2, phi_2_spline, True)

            raw_angles = np.zeros(12)
            raw_angles[0:3] = [-theta_0[0], -theta_1[0], -theta_2[0]] # BL
            raw_angles[3:6] = [theta_0[1], theta_1[1], theta_2[1]] # BR
            raw_angles[6:9] = [-theta_0[2], -theta_1[2], -theta_2[2]] # FL
            raw_angles[9:12] = [theta_0[3], theta_1[3], theta_2[3]] # FR
            
            data.ctrl[:] = cpg_core.clamp_to_joint_limits(raw_angles)

            mujoco.mj_step(model, data)
            current_height = float(data.body("base_link").xpos[2])
            height_deviation_sum += abs(current_height - flat_ground_height)
            height_samples += 1
                
        # evaluating fitness
        final_dx = data.body("base_link").xpos[0] - initial_pos[0]
        final_dy = data.body("base_link").xpos[1] - initial_pos[1]

        drift_penalty_weight = 2.0
        mean_height_deviation = (
            height_deviation_sum / height_samples if height_samples > 0 else 0.0
        )

        fitness = (
            final_dy
            - (drift_penalty_weight * abs(final_dx))
            - (HEIGHT_PENALTY_WEIGHT * mean_height_deviation)
        )
        return (
            robot_id,
            float(fitness),
            float(final_dx),
            float(final_dy),
            float(mean_height_deviation),
            False,
            "",
        )
        
    except Exception as e:
        # heavily penalized score so unstable genomes are eliminated
        return (robot_id, -999.0, 0.0, 0.0, 0.0, True, str(e))


def run_headless_pool(population: np.ndarray, max_workers: int = None) -> np.ndarray:
    if max_workers is None:
        max_workers = os.cpu_count()
        
    manager = mp.Manager()
    lock = manager.Lock()
    
    tasks = [(i, lock, population[i]) for i in range(len(population))]
    
    with mp.Pool(processes=max_workers) as pool:
        results = pool.map(simulate_universe, tasks)

    results.sort(key=lambda item: item[0])
    for robot_id, fitness, final_dx, final_dy, mean_height_deviation, failed, error in results:
        print(format_result_log(robot_id, final_dy, final_dx, mean_height_deviation, fitness, failed, error))

    fitness_scores = [result[1] for result in results]
    return np.array(fitness_scores)


def visualize_genome(genome: np.ndarray, sim_time: float, robot_id: int = 0) -> float:
    """
    Renders a single genome in a MuJoCo GUI window, calculates fitness, closes the window, and returns the score to the EA.
    """
    params = decode_genome(genome)
    
    xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
    flat_ground_height = get_flat_ground_base_height(xml_path)
    model = mujoco.MjModel.from_xml_path(xml_path)
        
    terrain_data = generate_blocky_terrain(nrow=model.hfield_nrow[0], ncol=model.hfield_ncol[0], verbose=True)
    model.hfield_data[:] = terrain_data
    data = mujoco.MjData(model)
    
    mujoco.mj_step(model, data)
    initial_pos = data.body("base_link").xpos.copy()
    height_deviation_sum = 0.0
    height_samples = 0
    current_height = float(data.body("base_link").xpos[2])
    height_deviation_sum += abs(current_height - flat_ground_height)
    height_samples += 1
    
    # CPG initialization
    dt = model.opt.timestep # integration timestep from XML
    omega = 0.25 # constant in the paper
    
    # actual amplitude, offset and phase variables for the 4 legs (BL, BR, FL, FR)
    target_offsets = np.array([0.0, 0.5, 0.25, 0.75]) * 2 * np.pi
    c_phi_0 = target_offsets.copy()
    
    c_a0, c_o0 = np.zeros(4), np.zeros(4)
    c_a1, c_o1 = np.zeros(4), np.zeros(4)
    c_a2_1, c_a2_2, c_o2 = np.zeros(4), np.zeros(4), np.zeros(4)
    
    # target params, np.full broadcasts across all 4 legs for symmetry
    t_a0 = np.full(4, params['mu_r0'])
    t_o0 = np.full(4, params['mu_o0'])
    t_a1 = np.full(4, params['mu_r1'])
    t_o1 = np.full(4, params['mu_o1'])
    t_a2_1 = np.full(4, params['mu_r2_1'])
    t_a2_2 = np.full(4, params['mu_r2_2'])
    t_o2 = np.full(4, params['mu_o2'])
    
    # static phase offsets for legs
    target_offsets = np.array([0.0, 0.5, 0.25, 0.75]) * 2 * np.pi
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time <= sim_time:
            step_start = time.time()
            # derivative updates for smooth stage change (eq 1, 2, 5, 6, 7)
            c_a0 = cpg_core.update_state_variables(c_a0, t_a0, params['gamma'], dt)
            c_o0 = cpg_core.update_state_variables(c_o0, t_o0, params['gamma'], dt)
            c_a1 = cpg_core.update_state_variables(c_a1, t_a1, params['gamma'], dt)
            c_o1 = cpg_core.update_state_variables(c_o1, t_o1, params['gamma'], dt)
            c_a2_1 = cpg_core.update_state_variables(c_a2_1, t_a2_1, params['gamma'], dt)
            c_a2_2 = cpg_core.update_state_variables(c_a2_2, t_a2_2, params['gamma'], dt)
            c_o2 = cpg_core.update_state_variables(c_o2, t_o2, params['gamma'], dt)

            # synchonize hips of all legs (eq 12)
            c_phi_0 = cpg_core.update_global_phases(c_phi_0, omega, params['coupling_w'], target_offsets, dt)
            # phase diff bw joints in the same leg
            phi_1, phi_2 = cpg_core.compute_intra_leg_phases(c_phi_0, params['psi_1'], params['psi_2'])

            # F_l filter
            phi_0_w = cpg_core.apply_duty_cycle_filter(c_phi_0, params['duty_cycle'])
            phi_1_w = cpg_core.apply_duty_cycle_filter(phi_1, params['duty_cycle'])
            phi_2_w = cpg_core.apply_duty_cycle_filter(phi_2, params['duty_cycle'])

            # amplitude selection for joint 2 and spline filter
            phi_2_2pi = np.mod(phi_2_w, 2 * np.pi) # selecting amplitude based on F_l filter from prev
            c_a2 = np.where(phi_2_2pi < np.pi, c_a2_1, c_a2_2) # switch between swing and stance amplitude
            phi_2_spline = cpg_core.apply_spline_filter(phi_2_w)

            theta_0 = cpg_core.compute_target_angles(c_a0, c_o0, phi_0_w, False)
            theta_1 = cpg_core.compute_target_angles(c_a1, c_o1, phi_1_w, False)
            theta_2 = cpg_core.compute_target_angles(c_a2, c_o2, phi_2_spline, True)

            raw_angles = np.zeros(12)
            raw_angles[0:3] = [-theta_0[0], -theta_1[0], -theta_2[0]] # BL
            raw_angles[3:6] = [theta_0[1], theta_1[1], theta_2[1]] # BR
            raw_angles[6:9] = [-theta_0[2], -theta_1[2], -theta_2[2]] # FL
            raw_angles[9:12] = [theta_0[3], theta_1[3], theta_2[3]] # FR
            
            data.ctrl[:] = cpg_core.clamp_to_joint_limits(raw_angles)
            
            # advance physics
            mujoco.mj_step(model, data)
            current_height = float(data.body("base_link").xpos[2])
            height_deviation_sum += abs(current_height - flat_ground_height)
            height_samples += 1
            viewer.sync()
            
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        # kill the viewer thread
        viewer.close() 

    # evaluating fitness
    final_dx = data.body("base_link").xpos[0] - initial_pos[0]
    final_dy = data.body("base_link").xpos[1] - initial_pos[1]

    drift_penalty_weight = 2.0
    mean_height_deviation = (
        height_deviation_sum / height_samples if height_samples > 0 else 0.0
    )

    fitness = (
        final_dy
        - (drift_penalty_weight * abs(final_dx))
        - (HEIGHT_PENALTY_WEIGHT * mean_height_deviation)
        )
    
    print(format_result_log(robot_id, final_dy, final_dx, mean_height_deviation, fitness))
    
    return float(fitness)


def run_visual_sequential(population: np.ndarray) -> np.ndarray:
    """
    Evaluates the population one by one in a GUI window.
    """
    fitness_scores = []
    for i, genome in enumerate(population):
        print(f"Launching Viewer for Robot {i}/{len(population)} ...")
        fit = visualize_genome(genome, sim_time=20.0, robot_id=i)
        fitness_scores.append(fit)
        
    return np.array(fitness_scores)

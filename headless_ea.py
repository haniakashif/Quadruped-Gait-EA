import os
import time
import numpy as np
import mujoco
import multiprocessing as mp
import kinematics as kin  

def generate_blocky_terrain(nrow, ncol):
    """Generates the 3-Zone Y-Oriented Testbed"""
    terrain = np.zeros((nrow, ncol))
    
    zone1_end = nrow // 3      # Flat
    zone2_end = 2 * nrow // 3  # Moderate
    
    # ZONE 2: MODERATE
    mod_block = 1
    mod_max_height = 0.5  
    for i in range(zone1_end, zone2_end, mod_block):
        for j in range(0, ncol, mod_block):
            terrain[i:i+mod_block, j:j+mod_block] = np.random.uniform(0.0, mod_max_height)

    # ZONE 3: CHALLENGING
    hard_block = 1
    hard_max_height = 1.0 
    for i in range(zone2_end, nrow, hard_block):
        for j in range(0, ncol, hard_block):
            terrain[i:i+hard_block, j:j+hard_block] = np.random.uniform(0.0, hard_max_height)
            
    return terrain.flatten()


def simulate_headless_robot(args):
    """The isolated physics loop that runs in each parallel universe"""
    robot_id, lock, theta_targets, steps_len, target_freq = args
    
    try:
        xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
        
        # Lock the compiler to prevent I/O race conditions on the hard drive
        with lock:
            model = mujoco.MjModel.from_xml_path(xml_path)
            
        # Generate and inject terrain
        terrain_data = generate_blocky_terrain(nrow=model.hfield_nrow[0], ncol=model.hfield_ncol[0])
        model.hfield_data[:] = terrain_data
        
        data = mujoco.MjData(model)
        
        # Step once to drop the robot and register initial coordinates
        mujoco.mj_step(model, data)
        initial_pos = data.body("base_link").xpos.copy()
        
        # Tracking variables
        max_sim_time = 20.0  # Run the evaluation for 20 simulated seconds
        last_print_time = 0.0
        
        while data.time <= max_sim_time:
            # Calculate trajectory index
            current_step = int(data.time * target_freq) % steps_len

            # Map the pre-computed angles to the motors
            data.ctrl[0:3]   = [theta_targets[2][0][current_step], theta_targets[2][1][current_step], theta_targets[2][2][current_step]]
            data.ctrl[3:6]   = [theta_targets[1][0][current_step], theta_targets[1][1][current_step], theta_targets[1][2][current_step]]
            data.ctrl[6:9]   = [theta_targets[3][0][current_step], theta_targets[3][1][current_step], theta_targets[3][2][current_step]]
            data.ctrl[9:12]  = [theta_targets[0][0][current_step], theta_targets[0][1][current_step], theta_targets[0][2][current_step]]

            # Advance physics
            mujoco.mj_step(model, data)
            
            # Distance tracking every 5.0 seconds
            if data.time - last_print_time >= 5.0:
                robot_pos = data.body("base_link").xpos
                dx = robot_pos[0] - initial_pos[0]  
                dy = robot_pos[1] - initial_pos[1]  
                total_distance = np.sqrt(dx**2 + dy**2)
                
                print(f"[Universe {robot_id:02d}] Sim Time: {data.time:.1f}s | Travelled: {total_distance:.4f}m")
                last_print_time += 5.0
                
        # Final evaluation return
        final_dx = data.body("base_link").xpos[0] - initial_pos[0]
        final_dy = data.body("base_link").xpos[1] - initial_pos[1]
        final_dist = np.sqrt(final_dx**2 + final_dy**2)
        
        return f"SUCCESS: Universe {robot_id:02d} completed 20s. Final Distance: {final_dist:.4f}m"
        
    except Exception as e:
        return f"FAILED: Universe {robot_id:02d} threw an error: {e}"


def main():
    population_size = 5
    target_freq = 50.0 
    
    print("Pre-computing Trajectory Matrix...")
    xyz = kin.generate_trajectory()
    
    xyz0 = kin.shift_trajectory(0, kin.rotate_trajectory(0, xyz)) # FR 
    xyz1 = kin.shift_trajectory(1, kin.rotate_trajectory(1, xyz)) # BR 
    xyz2 = kin.shift_trajectory(2, kin.rotate_trajectory(2, xyz)) # BL 
    xyz3 = kin.shift_trajectory(3, kin.rotate_trajectory(3, xyz)) # FL 

    theta_targets = [
        kin.inv_kin_array(xyz0, 0), # FR 
        kin.inv_kin_array(xyz1, 1), # BR 
        kin.inv_kin_array(xyz2, 2), # BL 
        kin.inv_kin_array(xyz3, 3)  # FL 
    ]
    steps_len = len(theta_targets[0][0])
    
    print(f"Spawning {population_size} parallel headless universes...")
    print("-" * 60)
    start_time = time.time()
    
    # Safely handle locks across CPU cores
    manager = mp.Manager()
    lock = manager.Lock()
    
    # Bundle the arguments (including the pre-computed arrays) for the worker pool
    tasks = [(i, lock, theta_targets, steps_len, target_freq) for i in range(1, population_size + 1)]
    
    # Launch the parallel workers
    with mp.Pool(processes=os.cpu_count()) as pool:
        results = pool.map(simulate_headless_robot, tasks)
        
    print("-" * 60)
    for result in results:
        print(result)
        
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Test Complete! {population_size} robots simulated 20s each in {total_time:.2f} real-world seconds.")

if __name__ == '__main__':
    main()
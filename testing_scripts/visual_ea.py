import os
# import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kinematics as kin  

def generate_blocky_terrain(nrow, ncol, seed):
    """Generates the 3-Zone Testbed with a locked seed for perfect reproducibility"""
    print(f"Generating Terrain for Universe {seed}...")
    np.random.seed(seed)  # Lock the random generation to match the headless ID
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


def main():
    population_size = 10
    target_freq = 50.0 
    
    print("Pre-computing Trajectory Matrix...")
    xyz = kin.generate_trajectory()
    xyz0 = kin.shift_trajectory(0, kin.rotate_trajectory(0, xyz)) 
    xyz1 = kin.shift_trajectory(1, kin.rotate_trajectory(1, xyz)) 
    xyz2 = kin.shift_trajectory(2, kin.rotate_trajectory(2, xyz)) 
    xyz3 = kin.shift_trajectory(3, kin.rotate_trajectory(3, xyz)) 

    theta_targets = [
        kin.inv_kin_array(xyz0, 0), kin.inv_kin_array(xyz1, 1), 
        kin.inv_kin_array(xyz2, 2), kin.inv_kin_array(xyz3, 3)
    ]
    steps_len = len(theta_targets[0][0])
    xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scene.xml")
    
    print(f"Launching Sequential GUI for {population_size} Iterations...")
    print("-" * 60)
    
    for iteration in range(1, population_size + 1):
        print(f"\n>>> LAUNCHING UNIVERSE {iteration:02d} <<<")
        
        # 1. Rebuild the universe from scratch for this iteration
        model = mujoco.MjModel.from_xml_path(xml_path)
        terrain_data = generate_blocky_terrain(nrow=model.hfield_nrow[0], ncol=model.hfield_ncol[0], seed=iteration)
        model.hfield_data[:] = terrain_data
        data = mujoco.MjData(model)
        
        mujoco.mj_step(model, data)
        initial_pos = data.body("base_link").xpos.copy()
        
        max_sim_time = 20.0
        last_print_time = 0.0
        
        # 2. Launch the viewer for this specific iteration
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time <= max_sim_time:
                step_start = time.time()

                current_step = int(data.time * target_freq) % steps_len

                data.ctrl[0:3]   = [theta_targets[2][0][current_step], theta_targets[2][1][current_step], theta_targets[2][2][current_step]]
                data.ctrl[3:6]   = [theta_targets[1][0][current_step], theta_targets[1][1][current_step], theta_targets[1][2][current_step]]
                data.ctrl[6:9]   = [theta_targets[3][0][current_step], theta_targets[3][1][current_step], theta_targets[3][2][current_step]]
                data.ctrl[9:12]  = [theta_targets[0][0][current_step], theta_targets[0][1][current_step], theta_targets[0][2][current_step]]

                mujoco.mj_step(model, data)
                
                # Print distance every 5 seconds
                if data.time - last_print_time >= 5.0:
                    robot_pos = data.body("base_link").xpos
                    dx = robot_pos[0] - initial_pos[0]  
                    dy = robot_pos[1] - initial_pos[1]  
                    total_distance = np.sqrt(dx**2 + dy**2)
                    
                    print(f"[Universe {iteration:02d}] Sim Time: {data.time:.1f}s | Travelled: {total_distance:.4f}m")
                    last_print_time += 5.0

                viewer.sync()
                
                # Artificial brake to make it viewable in real-time
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
            
            # Print final evaluation when 20s is hit, before opening the next window
            final_dx = data.body("base_link").xpos[0] - initial_pos[0]
            final_dy = data.body("base_link").xpos[1] - initial_pos[1]
            final_dist = np.sqrt(final_dx**2 + final_dy**2)
            print(f"--- Universe {iteration:02d} Complete. Final Distance: {final_dist:.4f}m ---")

if __name__ == '__main__':
    main()
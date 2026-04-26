import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import kinematics as kin  


def generate_blocky_terrain(nrow, ncol):
    print("Generating 3-Zone Testbed...")
    terrain = np.zeros((nrow, ncol))
    
    zone1_end = nrow // 3      # Flat: rows 0 to 33
    zone2_end = 2 * nrow // 3  # Moderate: rows 33 to 66
    
    # ZONE 1: FLAT
    # leave as zeros
    
    # ZONE 2: MODERATE
    mod_block = 1
    mod_max_height = 0.5  # 50% of max height
    
    for i in range(zone1_end, zone2_end, mod_block):
        for j in range(0, ncol, mod_block):
            random_height = np.random.uniform(0.0, mod_max_height)
            terrain[i:i+mod_block, j:j+mod_block] = random_height

    # ZONE 3: CHALLENGING
    hard_block = 1
    hard_max_height = 1.0 # 100% of max height
    
    for i in range(zone2_end, nrow, hard_block):
        for j in range(0, ncol, hard_block):
            random_height = np.random.uniform(0.0, hard_max_height)
            terrain[i:i+hard_block, j:j+hard_block] = random_height
            
    return terrain.flatten()


def main():
    print("Loading simulation...")
    xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scene.xml")
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        terrain_data = generate_blocky_terrain(nrow=model.hfield_nrow[0], ncol=model.hfield_ncol[0])
        model.hfield_data[:] = terrain_data
        data = mujoco.MjData(model)
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Pre-computing Trajectory...")
    xyz = kin.generate_trajectory()
    
    xyz0 = kin.shift_trajectory(0, kin.rotate_trajectory(0, xyz)) # FR (Leg 0)
    xyz1 = kin.shift_trajectory(1, kin.rotate_trajectory(1, xyz)) # BR (Leg 1)
    xyz2 = kin.shift_trajectory(2, kin.rotate_trajectory(2, xyz)) # BL (Leg 2)
    xyz3 = kin.shift_trajectory(3, kin.rotate_trajectory(3, xyz)) # FL (Leg 3)

    theta_targets = [
        kin.inv_kin_array(xyz0, 0), # FR (Index 0)
        kin.inv_kin_array(xyz1, 1), # BR (Index 1)
        kin.inv_kin_array(xyz2, 2), # BL (Index 2)
        kin.inv_kin_array(xyz3, 3)  # FL (Index 3)
    ]
    
    steps_len = len(theta_targets[0][0])
    target_freq = 50.0 # control loop freq
    print(f"Generated {steps_len} steps per cycle. Starting loop.")

    print("Launching Interactive Viewer.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # initial position
        mujoco.mj_step(model, data)
        initial_pos = data.body("base_link").xpos.copy()
        print(f"Spawn point (World Frame): X={initial_pos[0]:.4f}, Y={initial_pos[1]:.4f}, Z={initial_pos[2]:.4f}")
        
        try:
            while viewer.is_running():
                step_start = time.time()

                # Calculate which step we should be on based on the target frequency
                current_step = int(data.time * target_freq) % steps_len

                # order depends on the way defined in xml
                # BL Leg (Indices 0, 1, 2) <- theta_targets[2]
                data.ctrl[0] = theta_targets[2][0][current_step]
                data.ctrl[1] = theta_targets[2][1][current_step]
                data.ctrl[2] = theta_targets[2][2][current_step]

                # BR Leg (Indices 3, 4, 5) <- theta_targets[1]
                data.ctrl[3] = theta_targets[1][0][current_step]
                data.ctrl[4] = theta_targets[1][1][current_step]
                data.ctrl[5] = theta_targets[1][2][current_step]

                # FL Leg (Indices 6, 7, 8) <- theta_targets[3]
                data.ctrl[6] = theta_targets[3][0][current_step]
                data.ctrl[7] = theta_targets[3][1][current_step]
                data.ctrl[8] = theta_targets[3][2][current_step]

                # FR Leg (Indices 9, 10, 11) <- theta_targets[0]
                data.ctrl[9]  = theta_targets[0][0][current_step]
                data.ctrl[10] = theta_targets[0][1][current_step]
                data.ctrl[11] = theta_targets[0][2][current_step]

                mujoco.mj_step(model, data) # moce to next timestep
                
                robot_pos = data.body("base_link").xpos
                dx = robot_pos[0] - initial_pos[0]  
                dy = robot_pos[1] - initial_pos[1]  
                total_distance = np.sqrt(dx**2 + dy**2)
                
                print(f"Time: {data.time:.2f}s | Pos: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}) | ΔX: {dx:.4f}m | ΔY: {dy:.4f}m | Total: {total_distance:.4f}m")
                
                viewer.sync()

                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                    
        except KeyboardInterrupt:
            print("\nShutting down simulation...")

if __name__ == '__main__':
    main()
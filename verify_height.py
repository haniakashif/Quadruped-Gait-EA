import os
import time
import numpy as np
import mujoco
import mujoco.viewer

# --- LINUX WAYLAND FIX ---
# Forces GLFW to use X11 backend to prevent window destruction deadlocks
os.environ["GDK_BACKEND"] = "x11" 
os.environ["XDG_SESSION_TYPE"] = "x11"

SETTLE_STEPS = 100

def visualize_settle_height():
    print("Initializing MuJoCo environment for height verification...")
    
    # Load the scene
    xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    # 1. Force the terrain to be perfectly flat (Matches EA logic)
    model.hfield_data[:] = 0.0
    
    data = mujoco.MjData(model)
    
    # 2. Apply zero torque to all motors (Matches EA logic)
    data.ctrl[:] = 0.0
    
    dt = model.opt.timestep
    settled_height = None
    
    print(f"\nLaunching Viewer...")
    print(f"The robot will drop and settle under gravity.")
    print(f"The baseline height will be captured exactly at step {SETTLE_STEPS}.\n")

    # Launch the GUI
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_count = 0
        
        # Run for 5 seconds total so you have time to visually inspect its resting state
        while viewer.is_running() and data.time <= 5.0:
            step_start = time.time()
            
            # Advance physics
            mujoco.mj_step(model, data)
            viewer.sync()
            
            step_count += 1
            
            # 3. Capture the exact height at the SETTLE_STEPS mark
            if step_count == SETTLE_STEPS:
                settled_height = float(data.body("base_link").xpos[2])
                print(f"--> [SNAPSHOT TAKEN at step {SETTLE_STEPS}]")
                print(f"--> Calculated Base Link Z-Height: {settled_height:.4f} meters\n")

            # Sync to real-time clock so it doesn't instantly flash and close
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        
        viewer.close()
        
    print("Visualization complete.")
    if settled_height is not None:
        print(f"The mathematical flat ground height that the EA uses is: {settled_height:.4f} m")

if __name__ == "__main__":
    visualize_settle_height()
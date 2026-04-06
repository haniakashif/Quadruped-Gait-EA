import mujoco
import mujoco.viewer
import time
import os

def main():
    print("Loading simulation...")
    
    # Ensure it targets the correct scene file
    xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Launching Interactive Viewer. Press Ctrl+C in terminal or close window to exit.")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            while viewer.is_running():
                step_start = time.time()
                
                mujoco.mj_step(model, data)
                viewer.sync()
                
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                    
        except KeyboardInterrupt:
            # This triggers when you press Ctrl+C
            print("\nShutting down simulation ...")

if __name__ == '__main__':
    main()
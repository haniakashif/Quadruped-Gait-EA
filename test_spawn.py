import mujoco
import multiprocessing as mp
import time
import os

def spawn_robot_universe(args):
    robot_id, lock = args
    
    try:
        xml_path = os.path.join(os.path.dirname(__file__), "scene.xml")
        
        # lock the compiler so only one thread accesses the hard drive at a time
        with lock:
            model = mujoco.MjModel.from_xml_path(xml_path)
            
        data = mujoco.MjData(model)
        
        for _ in range(50):
            mujoco.mj_step(model, data)
            
        z_position = data.qpos[2]
        return f"SUCCESS: Universe {robot_id:03d} spawned. Current Z-Height: {z_position:.4f}m"
        
    except Exception as e:
        return f"FAILED: Universe {robot_id:03d} threw an error: {e}"

def main():
    population_size = 100
    
    print("Initializing Headless MuJoCo Test...")
    print(f"Spawning {population_size} parallel universes across 8 CPU threads...")
    print("-" * 60)
    
    start_time = time.time()
    
    # safely handle locks across CPU cores
    manager = mp.Manager()
    lock = manager.Lock()
    
    # bundle the Robot ID and the Lock together into a tuple for the worker function
    tasks = [(i, lock) for i in range(1, population_size + 1)]
    
    with mp.Pool(processes=8) as pool:
        results = pool.map(spawn_robot_universe, tasks)
        
    for result in results:
        print(result)
        
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Test Complete! 100 robots simulated in {total_time:.2f} seconds.")

if __name__ == '__main__':
    main()
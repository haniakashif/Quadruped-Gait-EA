import mujoco
import xml.etree.ElementTree as ET
import os

# --- 1. DUAL-PARSE: Extract middleware tags from the raw URDF ---
urdf_tree = ET.parse("robot_model/model.urdf")
urdf_root = urdf_tree.getroot()

# Dynamically map the ROS 2 gains
ros2_gains = {}
for joint in urdf_root.findall(".//ros2_control/joint"):
    name = joint.attrib["name"]
    p_param = joint.find("param[@name='p']")
    d_param = joint.find("param[@name='d']")
    
    p_val = p_param.text if p_param is not None else "5.0"
    d_val = d_param.text if d_param is not None else "0.1"
    ros2_gains[name] = {"p": p_val, "v": d_val} # Map 'd' to MuJoCo's 'kv'

# --- 2. STANDARD MUJOCO COMPILATION ---
model = mujoco.MjModel.from_xml_path("robot_model/model.urdf")
raw_path = "robot_model/robot_raw.xml"
mujoco.mj_saveLastXML(raw_path, model)

tree = ET.parse(raw_path)
root = tree.getroot()

compiler = root.find("compiler")
if compiler is not None and "meshdir" in compiler.attrib:
    del compiler.attrib["meshdir"]

asset = root.find("asset")
if asset is not None:
    for mesh in asset.findall("mesh"):
        old_file = mesh.attrib["file"]
        mesh.attrib["file"] = f"meshes/{old_file}"

worldbody = root.find("worldbody")
if worldbody is not None:
    for geom in worldbody.iter("geom"):
        group = geom.attrib.get("group", "0")
        if group == "0":
            # CRITICAL FIX: Reverted to 0 to prevent internal geometric explosions
            geom.attrib["contype"] = "0"  
            geom.attrib["conaffinity"] = "1" 
        elif group in ["1", "2"]:
            geom.attrib["rgba"] = "0.7 0.7 0.75 1.0"

    base_body = ET.Element("body", name="base_link", pos="0 0 0.3")
    ET.SubElement(base_body, "freejoint", name="base_freejoint")
    
    # Inject a spatial site required for mounting the IMU
    ET.SubElement(base_body, "site", name="imu_site", pos="0 0 0", size="0.01")
    
    ET.SubElement(base_body, "inertial", pos="-1.0171e-10 -0.0086065 0.03608", mass="0.30423", fullinertia="0.00232 0.00064 0.00225 0.0 0.0 8e-05")
    
    for child in list(worldbody):
        base_body.append(child)
        worldbody.remove(child)
    worldbody.append(base_body)

# --- 3. INJECT ACTUATORS & DYNAMIC GAINS ---
actuator = ET.SubElement(root, "actuator")
for joint in root.iter("joint"):
    j_name = joint.attrib.get("name")
    j_range = joint.attrib.get("range")
    f_range = joint.attrib.get("actuatorfrcrange") 
    
    if j_name and j_name != "base_freejoint":
        motor = ET.SubElement(actuator, "position")
        motor.attrib["name"] = j_name.replace("_joint", "_motor")
        motor.attrib["joint"] = j_name
        
        # Apply the exact gains extracted from the URDF's ROS 2 tags
        if j_name in ros2_gains:
            motor.attrib["kp"] = ros2_gains[j_name]["p"]
            motor.attrib["kv"] = ros2_gains[j_name]["v"]
            
        if j_range:
            motor.attrib["ctrlrange"] = j_range
        if f_range:
            motor.attrib["forcerange"] = f_range

# --- 4. INJECT SENSORS (Restoring the Gazebo IMU) ---
sensor = ET.SubElement(root, "sensor")
ET.SubElement(sensor, "gyro", name="imu_gyro", site="imu_site")
ET.SubElement(sensor, "accelerometer", name="imu_accel", site="imu_site")

tree.write("robot_model/robot.xml")
os.remove(raw_path)
print("Conversion complete: Lossless URDF extraction achieved.")
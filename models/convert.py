import mujoco
import xml.etree.ElementTree as ET
import os

model = mujoco.MjModel.from_xml_path("robot_model/model.urdf")
raw_path = "robot_model/robot_raw.xml"
mujoco.mj_saveLastXML(raw_path, model)

tree = ET.parse(raw_path)
root = tree.getroot()

# 1. Clean compiler tags and fix relative asset paths
compiler = root.find("compiler")
if compiler is not None and "meshdir" in compiler.attrib:
    del compiler.attrib["meshdir"]

asset = root.find("asset")
if asset is not None:
    for mesh in asset.findall("mesh"):
        old_file = mesh.attrib["file"]
        # mesh.attrib["file"] = f"robot_model/meshes/{old_file}"
        mesh.attrib["file"] = f"meshes/{old_file}"

# 2. Containerize the Robot, fix Self-Collision, paint Visuals
worldbody = root.find("worldbody")
if worldbody is not None:
    for geom in worldbody.iter("geom"):
        group = geom.attrib.get("group", "0")
        if group == "0":
            geom.attrib["contype"] = "1"  # 0 means disable self-collisions
            geom.attrib["conaffinity"] = "1"
        elif group == "1" or group == "2": # group 1 & 2 are other meshes, not collision meshes
            geom.attrib["rgba"] = "0.7 0.7 0.75 1.0"

    base_body = ET.Element("body", name="base_link", pos="0 0 0.3")
    ET.SubElement(base_body, "freejoint", name="base_freejoint")
    
    ET.SubElement(base_body, "inertial", pos="-1.0171e-10 -0.0086065 0.03608", mass="0.30423", fullinertia="0.00232 0.00064 0.00225 0.0 0.0 8e-05")
    
    for child in list(worldbody):
        base_body.append(child)
        worldbody.remove(child)
    worldbody.append(base_body)

# I think this part was added to fix the vibrating issue, but shouldn't be here
# 3. Inject Soft Contact Overrides
# default = root.find("default")
# if default is None:
#     default = ET.SubElement(root, "default")
# def_geom = default.find("geom")
# if def_geom is None:
#     def_geom = ET.SubElement(default, "geom")
# def_geom.attrib["solimp"] = "0.9 0.95 0.001"
# def_geom.attrib["solref"] = "0.02 1"

# 4. Generate the Actuators (Motors)
# This mimics the ros2_control position interface with p=5.0 and d=0.1
actuator = ET.SubElement(root, "actuator")
for joint in root.iter("joint"):
    j_name = joint.attrib.get("name")
    j_range = joint.attrib.get("range")
    f_range = joint.attrib.get("actuatorfrcrange") 
    
    if j_name and j_name != "base_freejoint":
        motor = ET.SubElement(actuator, "position")
        motor.attrib["name"] = j_name.replace("_joint", "_motor")
        motor.attrib["joint"] = j_name
        motor.attrib["kp"] = "5.0"   # Proportional gain (p)
        motor.attrib["kv"] = "0.1"   # Derivative gain (d)
        
        if j_range:
            motor.attrib["ctrlrange"] = j_range
        if f_range:
            motor.attrib["forcerange"] = f_range

tree.write("robot_model/robot.xml")
os.remove(raw_path)
print("Conversion complete!")
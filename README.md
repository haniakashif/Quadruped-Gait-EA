# Quadruped-Gait-EA
This repository hosts the code for evolving the gait of a quadruped robot in simulation using Mujoco and testing it in Gazebo

#### Local Ubuntu Setup

Open your terminal and run the following commands:

```bash

# Initialize and activate the virtual environment
python3 -m venv venv
source venv/bin/activate

# Install MuJoCo and parallel processing libraries
pip install mujoco numpy scipy
```

#### Repository Structure

The repository is structured in the following way:

*   `scene.xml` is setting the world with the robot, plane and the red stand as objects, and defines the friction and gravity in the world.

*   `view_robot.py` spawns one robot iteration with GUI enabled to see if its being spawned correctly, and can later be restuctured to view the best walking gait on the robot.

*   `test_spawn.py` spawns 100 robots in headless mode and this will be restructured to spawn the population of robots for the EA.

*   `convert.py` inside the `models` folder converts the exported model.urdf file of the robot to the .xml format which MuJoCo requires.

The cave is the final environment which needs to be tested, but we will first move onto setting up the EA pipeline and make it walk on flat terrain and then figure out the cave model.


#### Types of Motor Commands

In MuJoCo, the type of command a motor accepts is given by the `<actuator\>` tag defined in the XML. The motors strictly accept **target joint angles measured in radians**. 


#### The Control Pipeline (`data.ctrl`)

In the MuJoCo C-struct architecture, the entire state of the dynamic universe is stored in the `mjData` object. Within this object, there is a specific 1D contiguous memory array called `data.ctrl`. 

The control vector is a NumPy array and has length of the number of actuators defined in your XML. The index of the array **corresponds to the order the actuators were defined in the file**. 





#### Rendering Groups (`group="0"`, `group="1"`, `group="2"`)
*   **`group="0"` (Collision):** By convention, the MuJoCo compiler assigns all collision geometries to group 0.
*   **`group="1"` (Visual):** The compiler assigns all visual geometries (the detailed aesthetic meshes) to group 1. 
*   **`group="2"`:** This is often used as a fallback for secondary visual elements or decorative markers. 

We cater to `group="1"` and `group="2"` in the script to assign the `rgba="0.7 0.7 0.75 1.0"` color vector. Because `.STL` files are purely coordinate data without texture information, MuJoCo will render them completely invisible (or pitch black depending on the lighting) unless a material or RGBA vector is explicitly injected into the XML tag.

#### `contype` and `conaffinity` (Collision Filtering)

In MuJoCo, two objects will only physically collide if the following boolean logic returns `True`:
$( \text{contype}_1 \ \& \ \text{conaffinity}_2 ) > 0 \quad \text{OR} \quad ( \text{contype}_2 \ \& \ \text{conaffinity}_1 ) > 0$

By setting your robot's legs to `contype="0"` and `conaffinity="1"`:
*   **Self-Collision is Disabled:** If Leg A (`contype="0"`) approaches Leg B (`conaffinity="1"`), the bitwise AND operation evaluates to 0, and the physics engine ignores the contact.
*   **World Collision Remains Active:** Earlier, we set the floor to `contype="1"` and `conaffinity="0"`. If a robot foot (`conaffinity="1"`) touches the floor (`contype="1"`), the intersection is greater than 0. The solver registers the collision, allowing the robot to stand.


#### Inertia Matrix Mapping: URDF vs. MuJoCo

The inertia tensor is a 3x3 symmetric matrix:
$$I = \begin{bmatrix}
I_{xx} & I_{xy} & I_{xz} \\
I_{xy} & I_{yy} & I_{yz} \\
I_{xz} & I_{yz} & I_{zz}
\end{bmatrix}$$

*   **URDF Format:** Interleaves the diagonals and off-diagonals alphanumerically.
    `<inertia ixx="A" ixy="B" ixz="C" iyy="D" iyz="E" izz="F"/>`
*   **MuJoCo `fullinertia` Format:** Strictly groups the principal moments of inertia (diagonals) first, followed by the products of inertia (off-diagonals).
    `fullinertia="A D F B C E"` ($I_{xx}\ I_{yy}\ I_{zz}\ I_{xy}\ I_{xz}\ I_{yz}$)




























#### Constraints and Collisions Solver

MuJoCo uses a regularized "soft constraint" solver. It models every contact point as a microscopic, multidimensional spring-damper system.  Because the URDF cannot provide parameters for these springs, MuJoCo defaults to extremely stiff values. When your high-poly `.STL` foot hits the floor, these stiff springs fight each other, causing the high-frequency vibration you observed.

By injecting `solimp` (Solver Impedance) and `solref` (Solver Reference), we are explicitly tuning the stiffness and damping of these microscopic contact springs, allowing the simulation to smoothly absorb the jagged edges of the CAD mesh. This is actually *more* realistic than Gazebo's rigid approach, as real-world materials possess localized compliance (deformation).

Here is the line-by-line breakdown of the code and the explanation of how values are processed.

### Section 3: Injecting Soft Contact Overrides

```python
default = root.find("default")
if default is None:
    default = ET.SubElement(root, "default")
```

  * **Explanation:** The script searches the root XML tree for a `<default>` tag. In MuJoCo, the `<default>` block applies settings globally to every class in the simulation. If the tag does not exist in the parsed output, Python's `ElementTree` library (`ET`) creates it dynamically.

<!-- end list -->

```python
def_geom = default.find("geom")
if def_geom is None:
    def_geom = ET.SubElement(default, "geom")
```

  * **Explanation:** Inside the `<default>` block, it looks for a `<geom>` tag. This specifies that the subsequent rules should only apply to physical geometries (not joints or motors). If it is missing, the script creates it.

<!-- end list -->

```python
def_geom.attrib["solimp"] = "0.9 0.95 0.001"
def_geom.attrib["solref"] = "0.02 1"
```

  * **Explanation:** We inject the soft-constraint parameters as dictionary attributes.
      * `solimp`: Defines the impedance function (how the constraint force scales as bodies push into each other). The values `$0.9\ 0.95\ 0.001$` create a non-linear spring that gets exponentially stiffer the deeper the overlap.
      * `solref`: Defines the time constant and damping ratio. `$0.02\ 1$` tells the solver to recover from a penetration in 0.02 seconds with a critically damped response (no bouncing).

### Section 4: Generating the Actuators

```python
actuator = ET.SubElement(root, "actuator")
```

  * **Explanation:** Creates the `<actuator>` container, which is MuJoCo's native block for defining motors.

<!-- end list -->

```python
for joint in root.iter("joint"):
    j_name = joint.attrib.get("name")
    j_range = joint.attrib.get("range")
```

  * **Explanation:** The script recursively iterates through every `<joint>` tag in the compiled XML.
      * **How it takes from URDF:** The MuJoCo C-compiler already processed your URDF before this Python script runs. It took your URDF's \`\<limit lower="-1.5707" upper="1.5707"/\>` and internally converted it to the MuJoCo attribute `range="-1.5707 1.5707"`. The `.get()` method safely extracts this converted string.

```python
    if j_name and j_name != "base_freejoint":
```
*   **Explanation:** We must verify a name exists, and explicitly skip the `base_freejoint` we created earlier. If we attached a motor to the freejoint, the robot would magically fly through the air instead of using its legs.

```python
        motor = ET.SubElement(actuator, "position")
        motor.attrib["name"] = j_name.replace("_joint", "_motor")
        motor.attrib["joint"] = j_name
```
*   **Explanation:** Creates a native `<position>` actuator (a PD controller) and logically renames it (e.g., `fl_hip_joint` becomes `fl_hip_motor`), assigning it to control the respective joint.

### Why `kp`, `kv`, and `forcerange` are Hardcoded

```python
        motor.attrib["kp"] = "5.0"   # Proportional gain (p)
        motor.attrib["kv"] = "0.1"   # Derivative gain (d)
        if j_range:
            motor.attrib["ctrlrange"] = j_range
            motor.attrib["forcerange"] = "-0.9414 0.9414"
```

**1. The Missing Gain Values (`kp` and `kv`)**
You defined `p=5.0` and `d=0.1` inside your URDF. However, you placed them inside the `<ros2_control>` tags. [cite_start]Because ROS 2 is a middleware and not part of the standard URDF rigid-body schema, the MuJoCo C-parser is hardcoded to intentionally ignore and delete the entire `<ros2_control>` block to prevent memory corruption. 
Because the parser destroyed this data before our Python script could read it, we must hardcode $k_p$ and $k_v$ to ensure the MuJoCo PD controllers perfectly mimic your Gazebo PID configurations.

**2. The Force Range (`forcerange`)**
In your URDF, you defined `<limit effort="0.9414"/>`. The MuJoCo compiler read this and mapped it to `actuatorfrcrange` on the joint. While we *could* extract it dynamically from the joint like we did with the `range`, hardcoding it directly onto the motor's `forcerange` is a mathematically identical shortcut. It strictly caps the motor output to $\pm$ 0.9414 Nm, ensuring the simulated position controller cannot exert more torque than the real-world HS-645MG servo is physically capable of producing.



























#### Execution Within the EA Algorithm

For the EA pipeline, what we want to accomplish is optimizing the parameters of the gait function. Which means we will need to pass on some parameters (as defined by our function), which will be computing positions on the gait function, (I think it would work this way) and we pass that onto the inverse kinematics solver of the robot (which I have functions for when we get to it) so that it computes the necessary angles for the leg to reach that point. 

The walking cycle of the robot will be evaluated based on the fitness and we also have to figure out the rest of the standard EA stuff.

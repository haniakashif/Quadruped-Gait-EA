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


#### Execution Within the EA Algorithm

For the EA pipeline, what we want to accomplish is optimizing the parameters of the gait function. Which means we will need to pass on some parameters (as defined by our function), which will be computing positions on the gait function, (I think it would work this way) and we pass that onto the inverse kinematics solver of the robot (which I have functions for when we get to it) so that it computes the necessary angles for the leg to reach that point. 

The walking cycle of the robot will be evaluated based on the fitness and we also have to figure out the rest of the standard EA stuff.

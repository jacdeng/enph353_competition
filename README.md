# 2019F UBC Parking competition

The repository contains the following ROS packages:

| Folders         | Description      |
|:--------------- |:---------------- |
| enph353_gazebo  | describes simulation world |
| enph353_npcs    | describes and controls pedestrian and Ford truck |
| enph353_utils   | contains competition startup scripts |
| adeept_awr      | describes and controls simulated Adeept AWR robot |
| adeept_awr_ros_driver | controls real world Adeept AWR robot |

## Installation instructions:
* If you **do not** have a workspace already create one in your home directory.
```
mkdir -p ~/353_ws/src
cd ~/353_ws/src
```

* Clone the repository into a catkin workspace src folder.
```
git clone https://github.com/ENPH353/2019F_competition_students.git
```

* Build the packages
```
roscd; cd ..
catkin_make
```

* Source the environment
```
source devel/setup.bash
```

* Start the simulated world
```
cd src/enph353_2019F_competition/enph353/enph353_utils/scripts
./run_sim.sh -vpgl
```
The available options are:

| Option | Description      |
|:-------|:---------------- |
| -v     | spawn vehicle    |
| -p     | spawn pedestrian |
| -g     | generate new license plates |
| -l     | display QR code labels |
# enph353_competition

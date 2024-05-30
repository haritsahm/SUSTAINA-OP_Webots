<h1 align="center">
  SUSTAINA-OP&trade; Webots
</h1>
  <p align="center">
    Simulation Environment for SUSTAINA-OP&trade; RoboCup 2023 Edition in Webots
  </P>
  <p align="center">
    <table align="center">
      <tr>
        <td><img src="https://github.com/SUSTAINA-OP/SUSTAINA-OP-Webots-DEV/assets/53966390/94a7014d-bc2a-42d9-b485-f7e0c8c7f9cd" width="160px"></td>
        <td><img src="https://github.com/SUSTAINA-OP/SUSTAINA-OP-Webots-DEV/assets/53966390/4a8feaa1-0a69-42d2-83d5-299b3d0b139a" width="160px"></td>
      </tr>
      <tr>
        <th>Real Robot</th>
        <th>Webots Model</th>
      </tr>
    </table>
  </P>
      
> [!Note]
> SUSTAINA-OP&trade; is an open hardware platform and is based on real robots. Therefore, the idea of publishing this content is based on SUSTAINA-OP&trade;. Please confirm https://github.com/SUSTAINA-OP/SUSTAINA-OP before use.

## What is SUSTAINA-OP&trade; Webots?

This is the "webots R2023b" simulator environment of the open-source robot simulator [Webots](https://github.com/cyberbotics/webots) for the robot SUSTAINA-OP&trade; competed in RoboCup 2023.

## SUSTAINA-OP&trade; walking

A sample program of walking control using preview control.
You can send walking commands to move the robot and check the camera images mounted on the robot.

**Required emvironment**
- webots R2023b 
- python3.8 or later

**Execute**
```
cd /SUSTAINA-OP-Webots-DEV/webots/world
webots walking.wbt
```
**Sending walking direction**
```
python3 walk_client.py <target x> <target y> <target theta>
```
**Viewing image from camera**
```
python3 view_image.py
```

[For more information](https://github.com/SUSTAINA-OP/SUSTAINA-OP-Webots-DEV/blob/master/webots/controllers/SUSTAINA-OP_walking)

<img src="https://github.com/SUSTAINA-OP/SUSTAINA-OP-Webots-DEV/assets/53966346/4a2fd818-29d2-4836-8511-e22c04f7bed1" width="320px">

## License Information
### Hardware

Please confirm the [LICENSE-HW](/LICENSE-HW) for license information.

Hardware includes the following content.
- 3D mesh models
- Files containing robot hardware information (e.g. [/SUSTAINA-OP.urdf](/SUSTAINA-OP.urdf))

### Software

Please confirm the licence notice in each file for licence information.

Software includes the following content.
- Files in [/webots](/webots) directory

> [!IMPORTANT]
> When adapting the content developed in this simulation to a real robot, the SUSTAINA-OP&trade; RoboCup 2023 Edition should be used.

import logging
from controller import Robot, Motor, TouchSensor, PositionSensor
from kinematics_dynamics import KinematicsDynamics
from devices import Servo, Sensors, ForceSensor
from omegaconf import DictConfig
import hydra
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from walk_engines import OpenZMPWalk

logger = logging.getLogger(__file__)

class WalkController:
    """
    Omnidirectional walk controller based on ARTEMIS robot architecture
    Reference: ARTEMIS Thesis (2023)
    """
    def __init__(self, wb_robot: Robot, config: DictConfig):
        self.config = config

        # Webot placeholder
        self.wb_robot = wb_robot
        self.wb_joints: Dict[str, Motor] = {}
        self.wb_sensors: Dict[str, Any] = {}

        # Robot placeholder
        self.actuators: Dict[str, Servo] = {}
        self.sensors: Dict[str, Sensors] = {}

        # Time step in milliseconds
        self.controller_cycle_hz = int(self.config.controller.control_frequency)
        self.controller_time_s = float(1/self.controller_cycle_hz)

        logger.info("Starting walk controller with %f hz - %f s cycle loop", self.controller_cycle_hz, self.controller_time_s)

        # Initialize the robot
        self.initialize_robot_devices()

        # Update preview control dt
        self.config.preview_control.dt = self.controller_time_s
        self.config.foot_step_planner.dt = self.controller_time_s

        self.walk_engine = None
        if config.walk_engine == 'open_zmp_walk':
            self.walk_engine = OpenZMPWalk(self.config)
        else:
            raise ValueError(f"Unknown walk engine: {config.walk_engine}")

        self.robot_kd = KinematicsDynamics()
        self.robot_kd.initialize()

        self.walk_engine.reset_frames(self.robot_kd)

    def initialize_robot_devices(self):
        """Initialize Webots devices based on the configuration"""
        # Process each device group in the robot configuration
        for device_group in self.config.robot_config:
            for device_type, list_devices in device_group.items():
                for device in list_devices:
                    # Initialize servo motors
                    if device_type == "servo":
                        # Get the Webots motor device
                        motor: Optional[Motor] = self.wb_robot.getDevice(device.name)
                        if motor:
                            # Store the Webots motor object
                            self.wb_joints[device.name] = motor

                            motor_sensor_name = f"{device.name}_sensor"
                            motor_sensor = self.wb_robot.getDevice(motor_sensor_name)
                            if motor_sensor:
                                logger.info("Found motor %s sensor in Robot." , motor_sensor_name)
                                motor_sensor.enable(self.controller_cycle_hz)

                            # Create a Servo object as a data holder (not a wrapper)
                            self.actuators[device.name] = Servo(name=device.name, id=device.id)
                        else:
                            logger.error("Device %s not found in Webot.", device.name)

                    # Initialize touch sensors
                    elif device_type == "touch_sensor":
                        # Get the Webots touch sensor device
                        sensor: Optional[TouchSensor] = self.wb_robot.getDevice(device.name)
                        if sensor:
                            logger.info("Found %s sensor in Robot." , device.name)
                            # Store the Webots sensor object
                            self.wb_sensors[device.name] = sensor
                            
                            # Enable the sensor with default sampling period
                            sensor.enable(self.controller_cycle_hz)
                            
                            # Create a ForceSensor object as a data holder
                            self.sensors[device.name] = ForceSensor(device.name, 'force')
                        else:
                            logger.error("Device %s not found in Webot.", device.name)

    def update(self):
        """Main update loop for the walk controller"""
        self.get_joint_states()

        q_desired, qdot_desired, tau_desired = self.update_fsm()

        # 7. Send commands to hardware
        self.send_joints_command_to_hardware(q_desired, qdot_desired, tau_desired)
        
    def get_joint_states(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Get current joint positions and velocities
        
        Returns:
            Tuple containing dictionaries of joint positions and velocities
        """
        for name, motor in self.wb_joints.items():
            # Get position directly from the motor
            position = 0.0
            position_sensor: Optional[PositionSensor] = motor.getPositionSensor()
            if position_sensor:
                position = position_sensor.getValue()
            else:
                logger.error("Missing position sensor for join %s", name)

            # In Webots, we need to calculate velocity from position changes
            # This is a simplified implementation
            velocity = motor.getVelocity()
            torque = motor.getTorqueFeedback()
            self.actuators[name].update_state(position=position, velocity=velocity, torque=torque)

        for joint_name in self.actuators.keys():
            self.robot_kd.set_joint_angle(joint_id=self.actuators[joint_name].id, angle=self.actuators[joint_name].position)
        self.robot_kd.calc_forward_kinematics(0)

    def get_joint_torques(self) -> Dict[str, float]:
        """Get current joint torques (estimated)
        
        Returns:
            Dictionary of joint torques
        """
        # In Webots, torque is not directly available
        # This is a placeholder for future implementation
        torques = {name: 0.0 for name in self.wb_joints.keys()}
        return torques
    
    def get_estimated_contact_status(self, q: Dict[str, float], qdot: Dict[str, float], 
                         tau: Dict[str, float]) -> Dict[str, bool]:
        """Get contact status from force sensors

        
        Args:
            q: Joint positions
            qdot: Joint velocities
            tau: Joint torques
            
        Returns:
            Dictionary of contact statuses
        """
        contacts = {}
        
        # Process all touch sensors
        for name, sensor in self.wb_sensors.items():
            if 'touch_sensor' in name:
                # Get the current force value from the Webots sensor
                force_value = sensor.getValue()
                
                # Update the corresponding ForceSensor data holder
                if name in self.sensors:
                    self.sensors[name].update_state(force_value)
                    
                # Determine contact status (true if force is detected)
                contacts[name] = force_value > 0.0
                    
        return contacts
    
    def update_state_estimator(self, q: Dict[str, float], qdot: Dict[str, float], 
                              contacts: Dict[str, bool]) -> None:
        """Update state estimator with current measurements
        
        Args:
            q: Joint positions
            qdot: Joint velocities
            contacts: Contact statuses
        """
        # Update IMU sensor if available
        if 'inertial_unit' in self.wb_sensors:
            imu = self.wb_sensors['inertial_unit']
            if 'inertial_unit' in self.sensors:
                # Get roll-pitch-yaw values from the IMU
                rpy = imu.getRollPitchYaw()
                
                # In a real implementation, you would convert RPY to quaternion
                # For now, using a placeholder quaternion
                # The actual implementation would use something like:
                # from scipy.spatial.transform import Rotation
                # r = Rotation.from_euler('xyz', rpy)
                # quat = r.as_quat()  # [x, y, z, w] format
                # quat_w_first = [quat[3], quat[0], quat[1], quat[2]]  # [w, x, y, z] format
                
                # Update the IMUSensor data holder with the quaternion
                placeholder_quaternion = [1.0, 0.0, 0.0, 0.0]  # [w, x, y, z] format
                self.sensors['inertial_unit'].update_state(placeholder_quaternion)
    
    def prepare_fsm_args(self, q: Dict[str, float], qdot: Dict[str, float], 
                        contacts: Dict[str, bool]) -> Dict[str, Any]:
        """Prepare arguments for the finite state machine
        
        Args:
            q: Joint positions
            qdot: Joint velocities
            contacts: Contact statuses
            
        Returns:
            Dictionary of arguments for the FSM
        """
        return {
            'q': q,
            'qdot': qdot,
            'contacts': contacts,
            'time': self.wb_robot.getTime()
        }
    
    def update_fsm(self) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Update finite state machine to generate desired states

        Args:
            fsm_args: Arguments for the FSM

        Returns:
            Tuple of desired joint positions, velocities, and torques
        """

        self.walk_engine.update(self.robot_kd)

        # This is a placeholder for the actual FSM implementation
        q_desired = {name: self.robot_kd.get_joint_angle(joint_id=self.actuators[name].id) for name in self.wb_joints.keys()}
        qdot_desired = {name: 0.0 for name in self.wb_joints.keys()}
        tau_desired = {name: 0.0 for name in self.wb_joints.keys()}

        return q_desired, qdot_desired, tau_desired
    
    def send_joints_command_to_hardware(self, q_desired: Dict[str, float], 
                                       qdot_desired: Dict[str, float], 
                                       tau_desired: Dict[str, float]) -> None:
        """Send commands to hardware

        Args:
            q_desired: Desired joint positions
            qdot_desired: Desired joint velocities
            tau_desired: Desired joint torques
        """
        for name, position in q_desired.items():
            if name in self.wb_joints:
                motor = self.wb_joints[name]
                motor.setPosition(position)

                # Set velocity if specified
                if name in qdot_desired and qdot_desired[name] > 0:
                    motor.setVelocity(qdot_desired[name])


# This code has been moved into the WalkController class

@hydra.main(version_base=None, config_path=".", config_name="config")
def execute_walk(config: DictConfig):
    wb_robot = Robot()
    wb_sim_time_ms = int(wb_robot.getBasicTimeStep())

    controller = WalkController(wb_robot=wb_robot, config=config)

    while (wb_robot.step(controller.controller_cycle_hz)) != -1:
        controller.update()

if __name__ == '__main__':
    execute_walk()

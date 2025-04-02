import numpy as np
from typing import Tuple, List, Union, Optional

def get_transition_xyz(position_x: float, position_y: float, position_z: float) -> np.ndarray:
    """
    Create a 3D position vector from x, y, z coordinates.
    
    Parameters
    ----------
    position_x : float
        X-coordinate
    position_y : float
        Y-coordinate
    position_z : float
        Z-coordinate
        
    Returns
    -------
    np.ndarray
        3D position vector [x, y, z]
    """
    return np.array([position_x, position_y, position_z])

def get_rotation_x(angle: float) -> np.ndarray:
    """
    Create a 3x3 rotation matrix for rotation around the X axis.
    
    Parameters
    ----------
    angle : float
        Rotation angle in radians
        
    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle), np.cos(angle)]])

def get_rotation_y(angle: float) -> np.ndarray:
    """
    Create a 3x3 rotation matrix for rotation around the Y axis.
    
    Parameters
    ----------
    angle : float
        Rotation angle in radians
        
    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])

def get_rotation_z(angle: float) -> np.ndarray:
    """
    Create a 3x3 rotation matrix for rotation around the Z axis.
    
    Parameters
    ----------
    angle : float
        Rotation angle in radians
        
    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])

def get_rotation_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Create a 3x3 rotation matrix from roll, pitch, and yaw angles.
    
    This function composes rotations in the ZYX order (yaw-pitch-roll),
    which is the standard convention for robotics applications.
    
    Parameters
    ----------
    roll : float
        Roll angle in radians (rotation around X axis)
    pitch : float
        Pitch angle in radians (rotation around Y axis)
    yaw : float
        Yaw angle in radians (rotation around Z axis)
        
    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    rotation_x = get_rotation_x(roll)
    rotation_y = get_rotation_y(pitch)
    rotation_z = get_rotation_z(yaw)
    return rotation_z @ rotation_y @ rotation_x

def get_translation_4d(position_x: float, position_y: float, position_z: float) -> np.ndarray:
    """
    Create a 4x4 homogeneous transformation matrix with translation.
    
    Parameters
    ----------
    position_x : float
        X-coordinate of the translation
    position_y : float
        Y-coordinate of the translation
    position_z : float
        Z-coordinate of the translation
        
    Returns
    -------
    np.ndarray
        4x4 homogeneous transformation matrix with identity rotation
    """
    return np.array([[1, 0, 0, position_x],
                     [0, 1, 0, position_y],
                     [0, 0, 1, position_z],
                     [0, 0, 0, 1]])

def get_transformation_xyzrpy(position_x: float, position_y: float, position_z: float, 
                           roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Create a 4x4 homogeneous transformation matrix from position and orientation.
    
    Parameters
    ----------
    position_x : float
        X-coordinate of the translation
    position_y : float
        Y-coordinate of the translation
    position_z : float
        Z-coordinate of the translation
    roll : float
        Roll angle in radians (rotation around X axis)
    pitch : float
        Pitch angle in radians (rotation around Y axis)
    yaw : float
        Yaw angle in radians (rotation around Z axis)
        
    Returns
    -------
    np.ndarray
        4x4 homogeneous transformation matrix combining translation and rotation
    """
    transformation = get_translation_4d(position_x, position_y, position_z)
    transformation[:3, :3] = get_rotation_rpy(roll, pitch, yaw)
    return transformation

def get_inverse_transformation(transform: np.ndarray) -> np.ndarray:
    """
    Calculate the inverse of a homogeneous transformation matrix.
    
    This function computes the inverse of a 4x4 homogeneous transformation matrix
    by transposing the rotation part and applying the appropriate translation.
    
    Parameters
    ----------
    transform : np.ndarray
        4x4 homogeneous transformation matrix to invert
        
    Returns
    -------
    np.ndarray
        4x4 inverse homogeneous transformation matrix
    """
    boa = -transform[:3, 3]  # Negative of the original translation
    x = transform[:3, 0]     # First column of rotation matrix
    y = transform[:3, 1]     # Second column of rotation matrix
    z = transform[:3, 2]     # Third column of rotation matrix
    
    # Create inverse transformation matrix
    # Transpose of rotation matrix + dot product of negative translation with rotation columns
    inv_t = np.array([[x[0], x[1], x[2], np.dot(boa, x)],
                      [y[0], y[1], y[2], np.dot(boa, y)],
                      [z[0], z[1], z[2], np.dot(boa, z)],
                      [0, 0, 0, 1]])
    return inv_t

def get_inertia_xyz(ixx: float, ixy: float, ixz: float, iyy: float, iyz: float, izz: float) -> np.ndarray:
    """
    Create a 3x3 inertia matrix from the six independent inertia components.
    
    Parameters
    ----------
    ixx : float
        Moment of inertia around x-axis
    ixy : float
        Product of inertia for x and y axes
    ixz : float
        Product of inertia for x and z axes
    iyy : float
        Moment of inertia around y-axis
    iyz : float
        Product of inertia for y and z axes
    izz : float
        Moment of inertia around z-axis
        
    Returns
    -------
    np.ndarray
        3x3 symmetric inertia matrix
    """
    return np.array([[ixx, ixy, ixz],
                     [ixy, iyy, iyz],
                     [ixz, iyz, izz]])

def convert_rotation_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """
    Extract roll, pitch, yaw angles from a rotation matrix.
    
    This function extracts the Euler angles (roll, pitch, yaw) from a 3x3 rotation
    matrix using the ZYX convention (yaw-pitch-roll), which is standard in robotics.
    
    Parameters
    ----------
    rotation : np.ndarray
        3x3 rotation matrix
        
    Returns
    -------
    np.ndarray
        Array of [roll, pitch, yaw] angles in radians
    """
    rpy = np.zeros(3)
    rpy[0] = np.arctan2(rotation[2, 1], rotation[2, 2])  # Roll
    rpy[1] = np.arctan2(-rotation[2, 0], np.sqrt(rotation[2, 1]**2 + rotation[2, 2]**2))  # Pitch
    rpy[2] = np.arctan2(rotation[1, 0], rotation[0, 0])  # Yaw
    return rpy

def convert_rpy_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Create a rotation matrix from roll, pitch, and yaw angles.
    
    This function creates a 3x3 rotation matrix from Euler angles using the
    ZYX convention (yaw-pitch-roll), which is standard in robotics.
    
    Parameters
    ----------
    roll : float
        Roll angle in radians (rotation around X axis)
    pitch : float
        Pitch angle in radians (rotation around Y axis)
    yaw : float
        Yaw angle in radians (rotation around Z axis)
        
    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    rotation = get_rotation_z(yaw) @ get_rotation_y(pitch) @ get_rotation_x(roll)
    return rotation

def convert_rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert roll, pitch, yaw angles to a quaternion.
    
    This function first creates a rotation matrix from the Euler angles and then
    converts the rotation matrix to a quaternion representation.
    
    Parameters
    ----------
    roll : float
        Roll angle in radians (rotation around X axis)
    pitch : float
        Pitch angle in radians (rotation around Y axis)
    yaw : float
        Yaw angle in radians (rotation around Z axis)
        
    Returns
    -------
    np.ndarray
        Quaternion as a 4D vector [w, x, y, z] where w is the scalar part
    """
    rotation = convert_rpy_to_rotation(roll, pitch, yaw)
    quaternion = np.zeros(4)
    quaternion[0] = 0.5 * np.sqrt(1 + rotation[0, 0] + rotation[1, 1] + rotation[2, 2])  # w
    quaternion[1] = (rotation[2, 1] - rotation[1, 2]) / (4 * quaternion[0])  # x
    quaternion[2] = (rotation[0, 2] - rotation[2, 0]) / (4 * quaternion[0])  # y
    quaternion[3] = (rotation[1, 0] - rotation[0, 1]) / (4 * quaternion[0])  # z
    return quaternion

def convert_rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to a quaternion.
    
    This function converts a 3x3 rotation matrix to a quaternion representation
    using a common conversion formula.
    
    Parameters
    ----------
    rotation : np.ndarray
        3x3 rotation matrix
        
    Returns
    -------
    np.ndarray
        Quaternion as a 4D vector [w, x, y, z] where w is the scalar part
    """
    quaternion = np.zeros(4)
    quaternion[0] = 0.5 * np.sqrt(1 + rotation[0, 0] + rotation[1, 1] + rotation[2, 2])  # w
    quaternion[1] = (rotation[2, 1] - rotation[1, 2]) / (4 * quaternion[0])  # x
    quaternion[2] = (rotation[0, 2] - rotation[2, 0]) / (4 * quaternion[0])  # y
    quaternion[3] = (rotation[1, 0] - rotation[0, 1]) / (4 * quaternion[0])  # z
    return quaternion

def convert_quaternion_to_rpy(quaternion: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to roll, pitch, yaw angles.
    
    This function first converts the quaternion to a rotation matrix and then
    extracts the Euler angles from the rotation matrix.
    
    Parameters
    ----------
    quaternion : np.ndarray
        Quaternion as a 4D vector [w, x, y, z] where w is the scalar part
        
    Returns
    -------
    np.ndarray
        Array of [roll, pitch, yaw] angles in radians
    """
    # Convert quaternion to rotation matrix
    # Formula from quaternion [w, x, y, z] to rotation matrix
    rotation = np.array([[quaternion[0]**2 + quaternion[1]**2 - quaternion[2]**2 - quaternion[3]**2,
                          2 * (quaternion[1] * quaternion[2] - quaternion[0] * quaternion[3]),
                          2 * (quaternion[1] * quaternion[3] + quaternion[0] * quaternion[2])],
                         [2 * (quaternion[1] * quaternion[2] + quaternion[0] * quaternion[3]),
                          quaternion[0]**2 - quaternion[1]**2 + quaternion[2]**2 - quaternion[3]**2,
                          2 * (quaternion[2] * quaternion[3] - quaternion[0] * quaternion[1])],
                         [2 * (quaternion[1] * quaternion[3] - quaternion[0] * quaternion[2]),
                          2 * (quaternion[2] * quaternion[3] + quaternion[0] * quaternion[1]),
                          quaternion[0]**2 - quaternion[1]**2 - quaternion[2]**2 + quaternion[3]**2]])
    
    # Extract roll, pitch, yaw angles from the rotation matrix
    return convert_rotation_to_rpy(rotation)

def convert_quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to a rotation matrix.
    
    This function converts a quaternion representation to a 3x3 rotation matrix
    using the standard conversion formula.
    
    Parameters
    ----------
    quaternion : np.ndarray
        Quaternion as a 4D vector [w, x, y, z] where w is the scalar part
        
    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    # Convert quaternion [w, x, y, z] to rotation matrix
    rotation = np.array([[quaternion[0]**2 + quaternion[1]**2 - quaternion[2]**2 - quaternion[3]**2,
                          2 * (quaternion[1] * quaternion[2] - quaternion[0] * quaternion[3]),
                          2 * (quaternion[1] * quaternion[3] + quaternion[0] * quaternion[2])],
                         [2 * (quaternion[1] * quaternion[2] + quaternion[0] * quaternion[3]),
                          quaternion[0]**2 - quaternion[1]**2 + quaternion[2]**2 - quaternion[3]**2,
                          2 * (quaternion[2] * quaternion[3] - quaternion[0] * quaternion[1])],
                         [2 * (quaternion[1] * quaternion[3] - quaternion[0] * quaternion[2]),
                          2 * (quaternion[2] * quaternion[3] + quaternion[0] * quaternion[1]),
                          quaternion[0]**2 - quaternion[1]**2 - quaternion[2]**2 + quaternion[3]**2]])
    return rotation

def calc_hatto(matrix3d: np.ndarray) -> np.ndarray:
    """
    Calculate the skew-symmetric matrix (hat operator) of a 3D vector.
    
    This function computes the skew-symmetric matrix representation of a 3D vector,
    which is used in cross product calculations and rotation representations.
    
    Parameters
    ----------
    matrix3d : np.ndarray
        3D vector [x, y, z]
        
    Returns
    -------
    np.ndarray
        3x3 skew-symmetric matrix
    """
    hatto = np.array([[0, -matrix3d[2], matrix3d[1]],
                      [matrix3d[2], 0, -matrix3d[0]],
                      [-matrix3d[1], matrix3d[0], 0]])
    return hatto

def calc_rodrigues(hatto: np.ndarray, angle: float) -> np.ndarray:
    """
    Calculate rotation matrix using Rodrigues' formula.
    
    This function computes a rotation matrix from a skew-symmetric matrix (created by
    the hat operator on a rotation axis) and a rotation angle using Rodrigues' formula.
    
    Parameters
    ----------
    hatto : np.ndarray
        3x3 skew-symmetric matrix representing the rotation axis
    angle : float
        Rotation angle in radians
        
    Returns
    -------
    np.ndarray
        3x3 rotation matrix
    """
    rodrigues = np.eye(3) + hatto * np.sin(angle) + (hatto @ hatto) * (1 - np.cos(angle))
    return rodrigues

def convert_rot_to_omega(rotation: np.ndarray) -> np.ndarray:
    """
    Extract the angular velocity vector from a rotation matrix.
    
    This function extracts the equivalent axis-angle representation from a rotation matrix
    and returns the angular velocity vector (scaled rotation axis).
    
    Parameters
    ----------
    rotation : np.ndarray
        3x3 rotation matrix
        
    Returns
    -------
    np.ndarray
        3D angular velocity vector
    """
    eps = 1e-10  # Small value to handle numerical precision issues
    
    # Calculate the rotation angle from the trace of the rotation matrix
    alpha = (rotation[0, 0] + rotation[1, 1] + rotation[2, 2] - 1) / 2
    alpha_dash = np.abs(alpha - 1)
    
    # Check if the rotation is close to identity (no rotation)
    if alpha_dash < eps:
        return np.zeros(3)
    else:
        # Extract the rotation angle
        theta = np.arccos(alpha)
        
        # Extract the rotation axis components from the skew-symmetric part
        omega = np.array([rotation[2, 1] - rotation[1, 2],
                          rotation[0, 2] - rotation[2, 0],
                          rotation[1, 0] - rotation[0, 1]])
        
        # Scale the axis by the rotation angle
        return 0.5 * (theta / np.sin(theta)) * omega

def calc_cross(vector3d_a: np.ndarray, vector3d_b: np.ndarray) -> np.ndarray:
    """
    Calculate the cross product of two 3D vectors.
    
    Parameters
    ----------
    vector3d_a : np.ndarray
        First 3D vector
    vector3d_b : np.ndarray
        Second 3D vector
        
    Returns
    -------
    np.ndarray
        Cross product of the two vectors
    """
    return np.cross(vector3d_a, vector3d_b)

def calc_inner(vector3d_a: np.ndarray, vector3d_b: np.ndarray) -> float:
    """
    Calculate the dot product (inner product) of two 3D vectors.
    
    Parameters
    ----------
    vector3d_a : np.ndarray
        First 3D vector
    vector3d_b : np.ndarray
        Second 3D vector
        
    Returns
    -------
    float
        Dot product of the two vectors
    """
    return np.dot(vector3d_a, vector3d_b)

def get_pose_3d_from_transform_matrix(transform: np.ndarray) -> np.ndarray:
    """
    Extract 6D pose (position and orientation) from a transformation matrix.
    
    This function extracts the 3D position and roll-pitch-yaw angles from a
    4x4 homogeneous transformation matrix.
    
    Parameters
    ----------
    transform : np.ndarray
        4x4 homogeneous transformation matrix
        
    Returns
    -------
    np.ndarray
        6D pose vector [x, y, z, roll, pitch, yaw]
    """
    pose_3d = np.zeros(6)
    # Extract position (translation) from the transformation matrix
    pose_3d[:3] = transform[:3, 3]
    # Extract orientation (roll, pitch, yaw) from the rotation part
    pose_3d[3:] = convert_rotation_to_rpy(transform[:3, :3])
    return pose_3d
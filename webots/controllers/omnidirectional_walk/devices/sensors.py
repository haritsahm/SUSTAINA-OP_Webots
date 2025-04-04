from typing import Tuple, Union, List
import numpy as np
from scipy.spatial.transform import Rotation

class Sensors:
    def __init__(self, name: str, id: int = -1):
        self._name = name
        self._id = id

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return self._id

    def update_state(self, *args, **kwargs):
        raise NotImplementedError('Subclasses must implement update_state')

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self._name}, id={self._id})'

class IMUSensor(Sensors):
    def __init__(self, name: str):
        super().__init__(name)
        self._quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z)

    def update_state(self, quat: Union[List, Tuple, np.ndarray]):
        quat_array = np.array(quat)
        norm = np.linalg.norm(quat_array)
        if not np.isclose(norm, 1.0, atol=1e-3):
            raise ValueError(f'Quaternion must be normalized (norm={norm:.4f})')
        self._quaternion = quat_array

    @property
    def quaternion(self):
        return self._quaternion

    def get_euler_angles(self) -> np.ndarray:
        rotation = Rotation.from_quat(self._quaternion, scalar_first=True)
        return rotation.as_euler('zyx', degrees=False)

    def __repr__(self):
        return f'IMUSensor(name={self._name}, quaternion={self._quaternion})'

class ForceSensor(Sensors):
    def __init__(self, name: str, sensor_type: str = 'force'):
        super().__init__(name)
        self._sensor_type = sensor_type
        self._force_z = 0.0
        self._force_vector = (0.0, 0.0, 0.0)

    def update_state(self, force: Union[float, Tuple[float, float, float]]):
        if self._sensor_type == 'force':
            self._force_z = force
        else:
            self._force_vector = force

    @property
    def sensor_type(self):
        return self._sensor_type

    @property
    def force_z(self):
        return self._force_z

    @property
    def force_vector(self):
        return self._force_vector

    def get_force(self) -> Union[float, Tuple[float, float, float]]:
        return self._force_z if self._sensor_type == 'force' else self._force_vector

    def __repr__(self):
        return f'ForceSensor(name={self._name}, type={self._sensor_type}, force_z={self._force_z}, force_vector={self._force_vector})'

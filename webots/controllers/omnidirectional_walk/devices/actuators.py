class Actuators:
    def __init__(self, name: str, id: int):
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

class Servo(Actuators):
    def __init__(self, name: str, id: int):
        super().__init__(name, id)
        self._position = 0.0  # in radians
        self._torque = 0.0    # in Newton-meters
        self._velocity = 0.0  # in m/s

    @property
    def position(self):
        return self._position

    @property
    def torque(self):
        return self._torque

    @property
    def velocity(self):
        return self._velocity

    def update_state(self, position: float, torque: float, velocity: float):
        self._position = position
        self._torque = torque
        self._velocity = velocity

    def __repr__(self):
        return f'Servo(name={self._name}, id={self._id}, position={self._position:.2f} rad, torque={self._torque:.2f} Nm, velocity={self._velocity:.2f} m/s)'

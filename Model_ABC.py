from abc import ABCMeta, abstractmethod
import numpy as np

class ModelABC(metaclass=ABCMeta):
    """Abstract base class for ODE models.

    Raises:
        NotImplementedError: Abstract methods must be implemented.
    """
    @property
    def parameters(self) -> dict:
        """Get the model parameters.

        Returns:
            dict: dictionary of parameter keys and values.
        """
        assert self._parameters is not None, "Parameters not initialised."
        return self._parameters
    
    @parameters.setter
    def parameters(self, p) -> None:
        self._parameters = p
    
    _parameters : dict = None
    """Parameter backing field defaults to None."""
    
    @property
    def X0(self) -> np.array:
        """Initial conditions.

        Returns:
            np.array: initial conditions state array, shape=(nx,)
        """
        if self._X0 is None:
            ic = self.getDefaultInitialConditions()
        else:
            ic = self._X0
        return ic
    
    @X0.setter
    def X0(self, val : np.array):
        """Set the initial conditions for the integration.

        Args:
            val (np.array): initial state conditions, shape=(nx,)
        """
        self._X0 = val
    
    _X0 : np.array = None
    """Initial conditions property backing field"""
    
    @property
    def nu(self) -> int:
        """Get the size of the input vector.

        Returns:
            int: size of input vector
        """
        return len(self.getInputNames())
    
    @property
    def nx(self) -> int:
        """Get the size of the state vector.

        Returns:
            int: size of the state vector.
        """
        return len(self.getStateNames())
    
    @property
    def ny(self) -> int:
        """Get the size of the output vector.

        Returns:
            int: size of the output vector.
        """
        return len(self.getOutputNames())
    
    @abstractmethod
    def evaluate(self, 
                 x : np.array, 
                 u : np.array=np.empty,
                 t : float=np.nan):
        """Evaluate the system derivatives and outputs.

        Args:
            x (np.array): current system state, shape=(nx,).
            u (np.array, optional): current system inputs, shape=(nu,). Defaults to np.empty.
            t (float, optional): current time. Defaults to np.nan.

        Raises:
            NotImplementedError: Abstract methods must be implemented.

        Returns:
            tuple[np.array, np.array]: tuple (return1, return2)
            WHERE
            return1: xdot, system state derivatives, shape=(nx,)
            reutnr2: y, system outputs, shpae=(ny,)
        """
        raise NotImplementedError
    
    @abstractmethod
    def getDefaultInitialConditions(self) -> np.array:
        """Get the default initial conditions.

        Raises:
            NotImplementedError: Abstract method must be implemented.

        Returns:
            np.array: vector of initial states, shape=(nx,).
        """
        raise NotImplementedError
    
    @abstractmethod
    def getStateNames(self) -> tuple:
        """Get the state names.

        Raises:
            NotImplementedError: Abstract method must be implemented.

        Returns:
            tuple: sequence of string literals representing the state names.
        """
        raise NotImplementedError
    
    @abstractmethod
    def getOutputNames(self) -> tuple:
        """Get the output names.

        Raises:
            NotImplementedError: Abstract method must be implemented.

        Returns:
            tuple: sequence of string literals represeting the output names.
        """
        raise NotImplementedError
    
    @abstractmethod
    def getInputNames(self) -> tuple:
        """Get the input names.

        Raises:
            NotImplementedError: Abstract method must be implemented.

        Returns:
            tuple: sequence of string literals representing the input names.
        """
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def getDefaultParameters() -> dict:
        """Get the default parameters.

        Raises:
            NotImplementedError: Abstract method must be implemented.

        Returns:
            dict: dictionary of parameter keys to be referenced and values.
        """
        raise NotImplementedError
    
    def GetNamedValue(self, 
                      name : str, 
                      names : tuple, 
                      array : np.array=None):
        """Helper function to extract value from named array.

        Args:
            name (str): Name in names
            names (tuple): tuple of names
            array (np.array): index matched array to names

        Returns:
            tuple[float, int]: tuple (value, index)
            WHERE
            value: is the value in the array matched at index of name
            index: the index of name in names.
        """
        index = names.index(name)
        value = array[index] if array is not None else np.nan  
        return value, index

class Example(ModelABC):
    @property
    def parameters(self) -> dict:
        return self._parameters
    
    def evaluate(self, x: np.array, u: np.array = np.empty, t: float = np.nan) -> tuple:
        return np.zeros((self.nx, 1)), np.zeros((self.ny, 1))
    
    @staticmethod
    def getDefaultInitialConditions() -> np.array:
        return np.zeros([0, 0, 0, 0])
    
    @staticmethod
    def getStateNames() -> tuple:
        return (
            'x1', 'x2', 'x3', 'x4'
        )
    
    @staticmethod
    def getInputNames() -> tuple:
        return (
            'u1'
        )
        
    @staticmethod
    def getOutputNames() -> tuple:
        return (
            'y1', 'y2'
        )
    
    @staticmethod
    def getDefaultParameters() -> dict:
        return {
            'a': 100,
            'b': 200
        }
    

if __name__ == "__main__":
    test : Example = Example()
    
    print(test.nx)
    print(test.nu)
    print(test.ny)
    
    xdot, y = test.evaluate(test.getDefaultInitialConditions())
    print(xdot)
    print(y)
    
    print(test.getDefaultParameters().keys())
    print(test.parameters)
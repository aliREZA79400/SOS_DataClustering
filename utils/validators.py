class ValidType:
    def __init__(self, type_):
        self._type = type_
        
    def __set_name__(self, owner_clasds, prop_name):
        self.prop_name = prop_name
        
    def __set__(self, instance, value):
        if not isinstance(value, self._type):
            raise ValueError(f'{self.prop_name} must be of type '
                             f'{self._type.__name__}'
                            )
        instance.__dict__[self.prop_name] = value
        
    def __get__(self, instance, owner_class):
        if instance is None:
            return self
        else:
            return instance.__dict__.get(self.prop_name, None)

class Cluster_Number_Range:
    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        
    def __set_name__(self, owner_class, name):
        self.name = name
        
    def __set__(self, instance, value):
        if not isinstance(value, int):
            raise ValueError(f'{self.name} must be an int.')
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f'{self.name} must be at least {self.min_value}')
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f'{self.name} cannot exceed {self.max_value}')
        instance.__dict__[self.name] = value
        
    def __get__(self, instance, owner_class):
        if instance is None:
            return self
        else:
            return instance.__dict__.get(self.name, None)
        

def validate_integer(
        arg_name, arg_value, min_value=None, max_value=None,
        custom_min_message=None, custom_max_message=None
):
    """Validates that `arg_value` is an integer, and optionally falls within specific
    bounds.
    A custom override error message can be provided when min/max bounds are exceeded.

    Args:
        arg_name (str): the name of the argument (used in default error messages)
        arg_value (obj): the value being validated
        min_value (int): optional, specifies the minimum value (inclusive)
        max_value (int): optional, specifies the maximum value (inclusive)
        custom_min_message (str): optional, custom message when value is less
            than minimum
        custom_max_message (str): optional, custom message when value is greater
            than maximum

    Returns:
        None: no exceptions raised if validation passes

    Raises:
        TypeError: if `arg_value` is not an integer
        ValueError: if `arg_value` does not satisfy the bounds
    """
    if not isinstance(arg_value, int):
        raise TypeError(f'{arg_name} must be an integer.')

    if min_value is not None and arg_value < min_value:
        if custom_min_message is not None:
            raise ValueError(custom_min_message)
        raise ValueError(f'{arg_name} cannot be less than {min_value}')

    if max_value is not None and arg_value > max_value:
        if custom_max_message is not None:
            raise ValueError(custom_max_message)
        raise ValueError(f'{arg_name} cannot be greater than {max_value}')


def validate_str(
        arg_name , arg_value
):
    if not isinstance(arg_value, str):
        raise TypeError(f'{arg_name} must be an string.')
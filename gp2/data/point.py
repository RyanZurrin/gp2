from uuid import uuid4


class Point:
    """ A single data point
    """

    def __init__(self, value):
        """ Create a new Point
        Parameters
        ----------
        value : numpy.ndarray
            The data
        """
        self.id = str(uuid4())
        self.value = value

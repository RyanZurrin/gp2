from uuid import uuid4
from .point import Point


class Collection:
    """ A collection of DataPoints
    """

    def __init__(self):
        self.id = str(uuid4())
        self.data = {}

    def add(self, datapoint):
        """ Add a DataPoint to the collection

        Parameters
        ----------
        datapoint
            The DataPoint to add

        """
        self.data[datapoint.id] = datapoint.value

    def remove(self, datapoint):
        """ Remove a DataPoint from the collection

        Parameters
        ----------
        datapoint
            The DataPoint to remove
        """
        self.data.pop(datapoint.id)

    def get(self, index):
        """ Get a DataPoint from the specified index in the collection

        Parameters
        ----------
        index : int
            The index of the DataPoint to get

        Returns
        -------
        DataPoint
            The DataPoint at the specified index
        """
        return self.data[list(self.data.keys())[index]]

    def get_point(self, _id):
        """ Get a DataPoint from the specified id in the collection

        Parameters
        ----------
        _id : str
            The id of the DataPoint to get

        Returns
        -------
        DataPoint
            The DataPoint with the specified id
        """

        return self.data[_id]

    def shuffle(self):
        """ Shuffle the collection
        """
        import numpy as np
        all_ids = list(self.data.keys())
        np.random.shuffle(all_ids)

        shuffled_data = {}
        for k in all_ids:
            shuffled_data[k] = self.data[k]

        self.data = shuffled_data

    def to_array(self):
        """ Convert the collection to a numpy array
        """
        import numpy as np
        datasize = len(self.data.keys())
        datashape = self.get(0).shape
        datatype = self.get(0).dtype

        a = np.zeros(((datasize,) + datashape), dtype=datatype)
        ids = []

        id_list = list(self.data.keys())

        for i in range(datasize):
            uniqid = id_list[i]
            a[i] = self.data[uniqid]
            ids.append(uniqid)

        return a, ids

    @staticmethod
    def from_point_list(datalist):
        """ dataList needs to be a list of DataPoints

        Parameters
        ----------
        datalist : list
            A list of DataPoints
        """
        c = Collection()

        for i in datalist:
            c.add(i)

        return c

    @staticmethod
    def from_list(datalist, keys=None):
        """ datalist can be anything

        Parameters
        ----------
        datalist : list
            A list of DataPoints
        keys : list
            A list of keys to use for the DataPoints
        """
        c = Collection()

        for i, v in enumerate(datalist):

            p = Point(v)

            if keys:
                p.id = keys[i]

            c.add(p)

        return c

    def __str__(self):
        return str(self.data)

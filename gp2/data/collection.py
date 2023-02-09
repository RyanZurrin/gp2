import numpy as np
from uuid import uuid4
from .point import Point


class Collection:

    def __init__(self):
        self.id = str(uuid4())
        self.data = {}

    def add(self, datapoint):
        self.data[datapoint.id] = datapoint.value

    def remove(self, datapoint):
        self.data.pop(datapoint.id)
        
    def get(self, index):
        '''
        '''
        return self.data[list(self.data.keys())[index]]

    def get_point(self, id):

        return self.data[id]

    def shuffle(self):

        all_ids = list(self.data.keys())
        np.random.shuffle(all_ids)

        shuffled_data = {}
        for k in all_ids:
            shuffled_data[k] = self.data[k]

        self.data = shuffled_data
        

    def to_array(self):
        '''
        '''

        datasize = len(self.data.keys())
        datashape = self.get(0).shape
        datatype = self.get(0).dtype

        a = np.zeros(((datasize,)+datashape), dtype=datatype)
        ids = []

        id_list = list(self.data.keys())

        for i in range(datasize):
            uniqid = id_list[i]
            a[i] = self.data[uniqid]
            ids.append(uniqid)

        return a, ids

    @staticmethod
    def from_point_list(datalist):
        '''
        dataList needs to be a list of DataPoints
        '''
        c = Collection()

        for i in datalist:

            c.add(i)

        return c

    @staticmethod
    def from_list(datalist, keys=None):
        '''
        datalist can be anything
        '''
        c = Collection()

        for i,v in enumerate(datalist):

            p = Point(v)

            if keys:
                p.id = keys[i]

            c.add(p)

        return c

    def __str__(self):
        return str(self.data)


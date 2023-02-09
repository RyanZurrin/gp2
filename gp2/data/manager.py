from .collection import Collection
from .point import Point

import pickle

class Manager:

    def __init__(self):
        self.collections = {}
        self.names = {}

    def register(self, collection, name=None):

        # here we need to remove the old collection
        if name:
            for c in self.names:
                if self.names[c] == name:
                    old_collection_id = c
                    del self.collections[old_collection_id]
                    del self.names[c]
                    # print('Unregistered stale collection:', name)
                    break

        # register this collection + name if given
        self.collections[collection.id] = collection
        
        if name:
            self.names[collection.id] = name

    def get(self, name):

        # get name
        for c in self.names:
            if self.names[c] == name:
                return self.collections[c]

    @staticmethod
    def add(collection, datapoint):

        collection.add(datapoint)
        return collection

    @staticmethod
    def remove(collection, datapoint):

        collection.remove(datapoint)
        return collection

    @staticmethod
    def remove_and_add(old_collection, new_collection, datapoint):
        
        old_collection.remove(datapoint)
        new_collection.add(datapoint)

        return old_collection, new_collection

    def find(self, datapoint_id, show_names=False):

        collections = []

        # also support point objects
        if type(datapoint_id)==Point:
            datapoint_id = datapoint_id.id

        for c in self.collections:
            
            if datapoint_id in self.collections[c].data:
                collections.append(self.collections[c])

        # just lookup names if requested
        collections_with_names = {}

        if show_names:
            for i,c in enumerate(collections):

                if collections[i].id in self.names:
                    name = self.names[collections[i].id]
                else:
                    name = 'unknown'+str(i)
                collections_with_names[name] = collections[i]
            collections = collections_with_names


        return collections


    def save(self, filepath):

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):

        with open(filepath, 'rb') as f:
            m = pickle.load(f)

        return m

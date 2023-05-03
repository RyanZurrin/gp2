from .collection import Collection
from .point import Point

import pickle


class Manager:
    """ A class to manage collections of data."""

    def __init__(self):
        """ Initialize the manager."""
        self.collections = {}
        self.names = {}

    def register(self, collection, name=None):
        """ Register a collection with the manager.
        Parameters
        ----------
        collection : Collection
            The collection to register.
        name : str
            The name to register the collection with.
        """
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
        """ Get a collection by name.
        Parameters
        ----------
        name : str
            The name of the collection to get.
        Returns
        -------
        Collection
            The collection with the given name.
        """
        # get name
        for c in self.names:
            if self.names[c] == name:
                return self.collections[c]

    @staticmethod
    def add(collection, datapoint):
        """ Add a datapoint to a collection.
        Parameters
        ----------
        collection : Collection
            The collection to add the datapoint to.
        datapoint : Point
            The datapoint to add.
        Returns
        -------
        Collection
            The collection with the added datapoint.
        """
        collection.add(datapoint)
        return collection

    @staticmethod
    def remove(collection, datapoint):
        """ Remove a datapoint from a collection.
        Parameters
        ----------
        collection : Collection
            The collection to remove the datapoint from.
        datapoint : Point
            The datapoint to remove.
        Returns
        -------
        Collection
            The collection with the removed datapoint.
        """
        collection.remove(datapoint)
        return collection

    @staticmethod
    def remove_and_add(old_collection, new_collection, datapoint):
        """ Remove a datapoint from one collection and add it to another.
        Parameters
        ----------
        old_collection : Collection
            The collection to remove the datapoint from.
        new_collection : Collection
            The collection to add the datapoint to.
        datapoint : Point
            The datapoint to remove and add.
        Returns
        -------
        Collection
            The old collection with the removed datapoint.
        Collection
            The new collection with the added datapoint.
        """
        old_collection.remove(datapoint)
        new_collection.add(datapoint)

        return old_collection, new_collection

    def find(self, datapoint_id, show_names=False):
        """ Find a collection by datapoint id.
        Parameters
        ----------
        datapoint_id : int
            The id of the datapoint to find.
        show_names : bool
            Whether to return the collection names instead of the collections.
        Returns
        -------
        list
            The collections containing the datapoint.
        """
        collections = []

        # also support point objects
        if type(datapoint_id) == Point:
            datapoint_id = datapoint_id.id

        for c in self.collections:

            if datapoint_id in self.collections[c].data:
                collections.append(self.collections[c])

        # just lookup names if requested
        collections_with_names = {}

        if show_names:
            for i, c in enumerate(collections):

                if collections[i].id in self.names:
                    name = self.names[collections[i].id]
                else:
                    name = 'unknown' + str(i)
                collections_with_names[name] = collections[i]
            collections = collections_with_names

        return collections

    def save(self, filepath):
        """ Save the manager to a file.
        Parameters
        ----------
        filepath : str
            The filepath to save the manager to.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        """ Load a manager from a file.
        Parameters
        ----------
        filepath : str
            The filepath to load the manager from.
        Returns
        -------
        Manager
            The loaded manager.
        """
        with open(filepath, 'rb') as f:
            m = pickle.load(f)

        return m

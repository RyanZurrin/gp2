from dataManager import DataManager
import uuid

class toyData(DataManager):
        A = [1, 2, 3, 4, 5, 6, 7, 8]
        B = ["a", "b", "c", "d", "e"]

        def init_test(self):
            self.collA = self.initCollection(self.A)
            self.collB = self.initCollection(self.B)
            print(self.collA)
            print(self.collB)

        def testAdd(self):
            self.collB = self.addToCollection("g", self.collB)
            print("Updated: %s" % self.collB)

        def testRemove(self):
            dpId = str(list(self.collB.keys())[0])
            print(dpId)
            print(uuid.UUID(dpId))
            self.collB = self.removeFromCollection(uuid.UUID(dpId), self.collB)
            print("Updated: %s" % self.collB)

        def testRemoveAndAdd(self):
            dpId = str(list(self.collA.keys())[0])
            (self.collB, self.collA) = self.removeAndAddToCollection(uuid.UUID(dpId), self.collA, self.collB)
            print("Updated: %s, %s" % (self.collB, self.collA))


td = toyData()
td.init_test()
td.testAdd()
td.testRemove()
td.testRemoveAndAdd()
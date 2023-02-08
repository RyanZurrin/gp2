from uuid import uuid4

class Point:

    def __init__(self, value):
        self.id = str(uuid4())
        self.value = value

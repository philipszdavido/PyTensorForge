class Vector:
    def __init__(self, data):
        self.data = data

    def set(self, index, value):
        self.data[index] = value

    def get(self, index):
        return self.data[index]

    @staticmethod
    def load(self, data):
        self.data = data

    def dump(self):
        return self.data

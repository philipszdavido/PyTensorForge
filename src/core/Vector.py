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

    @staticmethod
    def Zero(self, data):
        vec = Vector(data)
        for i in vec.data:
            vec.set(i, 0)
        return vec

    def zero(self):
        for i in range(len(self.data)):
            self.data[i] = 0

    def log(self):
        for i in range(len(self.data)):
            print(self.data[i])
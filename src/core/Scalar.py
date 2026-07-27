class Scalar:
    def __init__(self):
        self.data = None

    def log(self):
        print(self.data)

    def one(self):
        self.data = 1
        return self

    def zero(self):
        self.data = 0
        return self

    def add(self, other):
        self.data = self.data + other.data
        return self
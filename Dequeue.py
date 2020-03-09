class deque:
    def __init__(self):
        self.capacity = 100
        self.i = 0
        self.j = 0
        self.data = [0]*self.capacity

    def is_empty(self):
        return self.i == self.j

    def __nonzero__(self):
        return self.i < self.j

    def append(self, element):
        if self.j == self.capacity - 10:
            self.data.extend([0]*self.capacity)
            self.capacity *= 2
        self.data[self.j] = element
        self.j += 1

    def popleft(self):
        data = self.data[self.i]
        self.i += 1
        return data

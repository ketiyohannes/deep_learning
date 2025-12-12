k = [1, 2, 3, 4, 5]

class MyClass:
    def __init__(self, k):
        self.k = k

    def __len__(self):
        return len(self.k)

    def __getitem__(self, index):
        return self.k[index] + 10

my_class = MyClass(k)
print(my_class[2])
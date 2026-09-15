class Base:
    def parameters(self):
        return []

    def flush(self):
        for p in self.parameters():
            p.grad = 0.0

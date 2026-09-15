from micrograd.layer import Layer
from micrograd.Base  import Base

class MLP(Base):
    def __init__(self, nin, nouts, learning_rate: float = 0.05, epochs: int = 100):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.losses        = []

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def forward(self, x_in):
        y_preds = [self(x) for x in x_in]
        return y_preds

    def step(self):
        for p in self.parameters():
            p.data -= self.learning_rate * p.grad

    def mse(self, yin, y_pred):
        mse = sum([(yout - ygt) ** 2 for ygt, yout in zip(yin, y_pred)])
        return mse / len(yin)

    def train(self, x_in, y_in, disable_print=False):

        for epoch in range(self.epochs):
            self.flush()
            predictions = self.forward(x_in)
            loss        = self.mse(y_in, predictions)
            self.losses.append(loss.data)
            if not disable_print: print(f">>> {epoch + 1} / {self.epochs} completed. MSE loss: {loss}")
            loss.backward()
            self.step()


    def get_losses(self):
        return self.losses

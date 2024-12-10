"""
Be sure you have minitorch installed in your Virtual Env.
>>> pip install -Ue .
"""

import minitorch
import time

# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, t: minitorch.Tensor) -> minitorch.Tensor:
        batch, in_size = x.shape
        return (
            self.weights.value.view(1, in_size, self.out_size)
            * x.view(batch, in_size, 1)
            ).sum(1).view(batch, self.out_size) + self.bias.value.view(1, self.out_size)


class Network(minitorch.Module):
    def __init__(self, hidden_layers: int):
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 2)

        Returns:
            Tensor: Output tensor of shape (batch_size, 1) after sigmoid activation
        """
        h = self.layer1.forward(x).relu()
        h = self.layer2.forward(h).relu()
        return self.layer3.forward(h).sigmoid()

def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)

class TensorTrain:
    def __init__(self, hidden_layers: int):
        self.hidden_layers = hidden_layers
        self.model = Network(self.hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y).view(data.N, 1)  # Ensure y has shape (N, 1)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            start_time = time.time()
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward pass
            out = self.model.forward(X)
            prob = (out * y) + ((out - 1.0) * (y - 1.0))

            loss = -prob.log()
            loss_sum = (loss / data.N).sum().view(1)
            loss_sum.backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update parameters
            optim.step()

            # End timing
            epoch_time = time.time() - start_time

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y_pred = (out.detach() > 0.5).view(data.N)
                correct = int((y_pred == y.view(data.N)).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)

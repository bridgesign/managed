import torch
from managed import ManagedModule as mm
from concurrent.futures import ThreadPoolExecutor

# Inherit ManagedModule instead of torch.nn.Module
class MyModel(mm):
    """
    A simple model that takes in a tensor and returns the sum of its elements.
    """
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x).sum()

def run_experiment(trainloader):
    """
    Runs an experiment on the given trainloader.
    """
    model = MyModel().cuda()
    model.pin() # Pin the model - this is important!
    optimizer = torch.optim.Adam(model.parameters())
    while True:
        net_loss = 0
        for x, y in trainloader:
            optimizer.zero_grad()
            loss = model(x) - y
            net_loss += loss.item()
            loss.backward()
            optimizer.step()
        if net_loss < 1e-3:
            break
    return model

if __name__ == "__main__":
    executor = ThreadPoolExecutor(max_workers=4)
    trainloader = torch.utils.data.DataLoader(torch.randn(100, 10), batch_size=10)
    futures = []
    for _ in range(100):
        futures.append(executor.submit(run_experiment, trainloader))
    for future in futures:
        model = future.result()
        print(model(torch.randn(10)))

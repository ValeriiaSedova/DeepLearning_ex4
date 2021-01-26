import torch

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.layers = [
            torch.nn.Conv2d(64, 7, 2),
            torch.nn.BatchNorm(),
            torch.nn.ReLU(),
            torch.nn.MaxPool(3, 2),
            torch.nn.ResBlock(64, 1),
            torch.nn.ResBlock(128, 2),
            torch.nn.ResBlock(256, 2),
            torch.nn.ResBlock(512, 2),
            torch.nn.GlobalAvgPool(),
            torch.nn.Flatten(),
            torch.nn.Linear(2),
            torch.nn.Sigmoid()]

    def forward(self, input_tensor):
        x = input_tensor
        for layer in self.layers:
            x = layer.forward(x)
        return x


 
class ResBlock:

    def __init__(self, output_channels, stride):

        self.layers =[
            torch.nn.Conv2d(output_channels, 3, stride),
            torch.nn.BatchNorm(),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_channels, 1),
            torch.nn.BatchNorm(),
            torch.nn.ReLU()]


    def forward(self, input_tensor):
        x = input_tensor
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
        



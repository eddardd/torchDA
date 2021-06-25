import torch
from architectures.utils import LambdaLayer, GradReverse


class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2),
            torch.nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5),
            torch.nn.ReLU()
        )

        self.class_discriminator = torch.nn.Sequential(
            torch.nn.Linear(in_features=48 * 8 * 8, out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100, out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100, out_features=10),
            torch.nn.Softmax()
        )

        self.domain_discriminator = torch.nn.Sequential(
            LambdaLayer(GradReverse.apply),
            torch.nn.Linear(in_features=48 * 8 * 8, out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100, out_features=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        y = self.feature_extractor(x)
        features = y.view(-1, 48 * 8 * 8)
        predicted_labels = self.class_discriminator(features)
        predicted_domain = self.domain_discriminator(features)

        return predicted_labels, predicted_domain
import torch
import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        
        #self.conv1 = nn.Conv2d(3, 16, 3, padding=1) # 224 X 224 -->
        #self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        #self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        # max pooling layer
        #self.pool = nn.MaxPool2d(2, 2)
        #self.fc1 = nn.Linear(28 * 28 * 64, 256)
        #self.fc2 = nn.Linear(256, num_classes) # n_classes = 50       
        #self.dropout = nn.Dropout(p=dropout)        
        #self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)        
        #self.batch_norm2d = nn.BatchNorm2d(32)
        #self.batch_norm1d = nn.BatchNorm1d(256)
        
        self.features = nn.Sequential(
            #3,224,224
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #64,112,112
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #128,56,56
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            #256,28,28
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            #256,14,14
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
            #512,7,7
        )
            
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        ## Define forward behavior
        # sequence of convolutional and max pooling layers
        #x = self.pool(self.leaky_relu(self.conv1(x)))
        #x = self.pool(self.leaky_relu(self.conv2(x)))
        #x = self.batch_norm2d(x)
        #x = self.pool(self.leaky_relu(self.conv3(x)))        
        #x = x.view(-1, 28 * 28 * 64)        
        #x = self.dropout(x)       
        #x = self.leaky_relu(self.fc1(x))        
        #x = self.batch_norm1d(x)       
        #x = self.dropout(x)     
        #x = self.fc2(x)
        x=self.features(x)
        x = x.view(-1,512*7*7)
        x=self.linear_layers(x)
        return x

# instantiate the CNN
#model_scratch = MyModel()

# move tensors to GPU if CUDA is available
#if torch.cuda.is_available():
#  model_scratch.cuda()

######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"

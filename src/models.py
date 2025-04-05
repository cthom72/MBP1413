import torch.nn as nn

### models.py

class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=17, features=64):
        """
        Implements the DnCNN architecture from "Beyond a Gaussian Denoiser: Residual Learning 
        of Deep CNN for Image Denoising" (Zhang et al., 2017)
        
        Args:
            channels (int): Number of input/output channels (1 for grayscale, 3 for color)
            num_of_layers (int): Depth of the network
            features (int): Number of feature maps in each layer
        """
        super(DnCNN, self).__init__()
        
        # frst layer: Conv + ReLU
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # middle layers: Conv + BN + ReLU
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_of_layers - 2):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(features),
                    nn.ReLU(inplace=True)
                )
            )
        
        # last layer -> Conv 
        self.last_layer = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=3, padding=1, bias=False)
        
        # initialize weights according to literature
        self._initialize_weights()
    
    # feed forward function
    def forward(self, x):

        # store input for residual connection
        residual_input = x
        
        # first layer
        out = self.first_layer(x)
        
        # hidden layers
        for layer in self.hidden_layers:
            out = layer(out)
        
        # last layer
        out = self.last_layer(out)
        
        return residual_input - out
    
    # init weights using Kaiming initialization (in accordance with paper)
    def _initialize_weights(self):

        # iterate over modules
        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                # initialize convolution layers with Kaiming initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):

                # initialize the BatchNorm layers
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

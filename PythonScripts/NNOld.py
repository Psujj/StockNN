import torch
import torch.nn as nn


class StockPredictor(nn.Module):
    def __init__(self):
        super().__init__()


        # 1D Convolutional layers to process the 14-day time series (2 features per day)
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
       
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()


        # Flatten and fully connected layers to make the final prediction
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 14, 24)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(24, 1)


        # Loss function (mean squared error)
        self.criterion = nn.MSELoss()


    def forward(self, x):
        # Input x: shape (14, 2) → we need to change it to (1, 2, 14)
        x = x.transpose(0, 1).unsqueeze(0)  # (14, 2) → (1, 2, 14)
       
        # Pass through convolution layers
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))


        # Pass through fully connected layers
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)


        return x
   
    def loss(self, y_pred, y_true, prev_close):
        lp2Norm = self.criterion(y_pred, y_true)
        prev_close = prev_close[-1][0] # grab last closing value
        actual_chage = y_true - prev_close
        predicted_change = y_pred - prev_close


        direction = actual_chage * predicted_change
        penalty = (direction < 0).float()
        return lp2Norm + penalty * 0.05


        # penalty = torch.sigmoid(-actual_chage * predicted_change * 10)  # sharper sigmoid
        # return lp2Norm + 0.01 * penalty
        # hard threshold:
        # direction = actual_chage * predicted_change
        # if direction < 0:
        #     direction = 1
        # else :
        #     direction = 0
        # return lp2Norm + (direction * .05)
    # the l2 norm has a cost of about .003 - .0001 (.0001 comes from 1000 training cycles)




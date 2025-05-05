import torch
import torch.nn as nn

class StockPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(14 * 2, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 3)
        self.fc4 = nn.Linear(3, 1)

        self.dropout = nn.Dropout(0.1)
        self.RReLU = nn.RReLU()

        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = x.reshape(x.size(0), -1)

        x = self.RReLU(self.fc1(x))
        x = self.dropout(x)

        x = self.RReLU(self.fc2(x))
        x = self.dropout(x)
    
        x = self.RReLU(self.fc3(x))

        return self.fc4(x)
    
    def loss(self, y_pred, y_true, prev_close):
        lp2Norm = self.criterion(y_pred, y_true)
        prev_close = prev_close[-1][0] # grab last closing value
        actual_chage = y_true - prev_close
        predicted_change = y_pred - prev_close

        penalty = torch.sigmoid(-actual_chage * predicted_change * 10)  # sharper sigmoid
        return (lp2Norm + 0.01 * penalty) * 1000
    
        # hard threshold:
        # direction = actual_chage * predicted_change
        # if direction < 0:
        #     direction = 1
        # else :
        #     direction = 0
        # return lp2Norm + (direction * .05)

    # the l2 norm has a cost of about .003 - .0001 (.0001 comes from 1000 training cycles)
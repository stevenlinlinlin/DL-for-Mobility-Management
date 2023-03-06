import torch
import torch.nn as nn


# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.layer_num = layer_num

        self.lstm = nn.LSTM(input_size, hidden_size,
                            layer_num, batch_first=True, dropout=0.2)
        #self.lstm1 = nn.LSTMCell(self.input_size, self.hidden_size)
        #self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        #self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.fc1 = nn.Linear(hidden_size, input_size)
        #self.fc2 = nn.Linear(108, 2)

        #self.relu = nn.ReLU()

    #forward
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_num, x.size(
            0), self.hidden_size).requires_grad_().to(device)
        # Initialize cell state
        c0 = torch.zeros(self.layer_num, x.size(
            0), self.hidden_size).requires_grad_().to(device)
        #h1 = torch.zero(x.size(0), self.hidden_size, dtype=torch.float32).to(device)
        #c1 = torch.zero(x.size(0), self.hidden_size, dtype=torch.float32).to(device)
        #h2 = torch.zero(x.size(0), self.hidden_size, dtype=torch.float32).to(device)
        #c2 = torch.zero(x.size(0), self.hidden_size, dtype=torch.float32).to(device)
        #h3 = torch.zero(x.size(0), self.hidden_size, dtype=torch.float32).to(device)
        #c3 = torch.zero(x.size(0), self.hidden_size, dtype=torch.float32).to(device)

        # weights initialization
      #torch.nn.init.xavier_normal_(h1)
      #torch.nn.init.xavier_normal_(c1)
      #torch.nn.init.xavier_normal_(h2)
      #torch.nn.init.xavier_normal_(c2)
        #torch.nn.init.xavier_normal_(h3)
        #torch.nn.init.xavier_normal_(c3)

        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc1(out[:, -1, :])
        #out = self.fc2(out)
        return out


# GRU model
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_num):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num

        self.gru = nn.GRU(input_dim, hidden_dim, layer_num, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 3)
        self.relu = nn.ReLU()

    #forward
    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_num, x.size(
            0), self.hidden_dim)).requires_grad_().to(device)
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(self.relu(out[:, -1, :]))
        #out = hn.view(-1, self.hidden_dim)
        #out = self.fc(out)
        return out



# TCN model
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, c_in, num_channels):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(c_in, num_channels)
        #self.gap = GAP1d()
        self.dropout = nn.Dropout(0.2)
        #self.linear = nn.Linear(layers[-1],c_out)
        #self.init_weights()
        self.decoder = nn.Linear(num_channels[-1], 3)

    #def init_weights(self):
        #self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        return self.decoder(self.dropout(self.tcn(x)[:, :, -1]))
        #x = self.tcn(x)
        #x = self.gap(x)
        #x = self.dropout(x)
        #return self.linear(x)

#param x: size of (Batch, input_channel, seq_len)
#return: size of (Batch, output_channel, seq_len)

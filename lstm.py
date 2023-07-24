import torch
from torch import nn
import torch.nn.functional as F

class LSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(LSTMCell, self).__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.forget_input, self.forget_hidden, self.forget_bias = self.create_gate_parameters()
        self.input_input, self.input_hidden, self.input_bias = self.create_gate_parameters()
        self.output_input, self.output_hidden, self.output_bias = self.create_gate_parameters()
        self.cell_input, self.cell_hidden, self.cell_bias = self.create_gate_parameters()

    def create_gate_parameters(self):
        input_weights = nn.Parameter(torch.zeros(self.input_dim, self.hidden_dim))
        hidden_weights = nn.Parameter(torch.zeros(self.hidden_dim, self.hidden_dim))
        nn.init.xavier_uniform_(input_weights)
        nn.init.xavier_uniform_(hidden_weights)
        bias = nn.Parameter(torch.zeros(self.hidden_dim))
        return input_weights, hidden_weights, bias

    def forward(self, x, h, c):
        # x has shape [batch_size, seq_len, input_size]
        output_hiddens, output_cells = [], []
        for i in range(x.shape[1]):
            forget_gate = F.sigmoid((x[:, i] @ self.forget_input) + (h @ self.forget_hidden) + self.forget_bias)
            input_gate = F.sigmoid((x[:, i] @ self.input_input) + (h @ self.input_hidden) + self.input_bias)
            output_gate = F.sigmoid((x[:, i] @ self.output_input) + (h @ self.output_hidden) + self.output_bias)
            input_activations = F.tanh((x[:, i] @ self.cell_input) + (h @ self.cell_hidden) + self.cell_bias)
            c = (forget_gate * c) + (input_gate * input_activations)
            h = F.tanh(c) * output_gate
            output_hiddens.append(h.unsqueeze(1))
            output_cells.append(c.unsqueeze(1))
        return torch.concat(output_hiddens, dim=1), torch.concat(output_cells, dim=1)



class MultiLayerLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(MultiLayerLSTM, self).__init__()
        self.input_dim, self.hidden_dim, self.num_layers = input_dim, hidden_dim, num_layers
        self.layers = nn.ModuleList()
        self.layers.append(LSTMCell(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(LSTMCell(hidden_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, input_dim)
        nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)

    def forward(self, x, h):
        # x has shape [batch_size, seq_len, embed_dim]
        # h is a tuple containing h and c, each have shape [layer_num, batch_size, hidden_dim]
        hidden, cell = h
        output_hidden, output_cell = self.layers[0](x, hidden[0], cell[0])
        new_hidden, new_cell = [output_hidden[:, -1].unsqueeze(0)], [output_cell[:, -1].unsqueeze(0)]
        for i in range(1, self.num_layers):
            output_hidden, output_cell = self.layers[i](self.dropout(output_hidden), hidden[i], cell[i])
            new_hidden.append(output_hidden[:, -1].unsqueeze(0))
            new_cell.append(output_cell[:, -1].unsqueeze(0))
        return self.linear(self.dropout(output_hidden)), (torch.concat(new_hidden, dim=0), torch.concat(new_cell, dim=0))
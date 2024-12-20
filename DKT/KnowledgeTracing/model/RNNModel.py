import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

class DKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(DKT, self).__init__()
        # Definir las dimensiones de entrada, oculta y de salida
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True,nonlinearity='tanh')
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # Inicializar el estado oculto con ceros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        # Propagar los datos a través de la red
        out,hn = self.rnn(x, h0)
        # Pasar la salida a través de la capa lineal y la función de activación sigmoide
        res = self.sig(self.fc(out))
        return res
 
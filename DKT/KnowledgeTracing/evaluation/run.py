import sys
sys.path.append('../')
from model.RNNModel import DKT
from data.dataloader import getTrainLoader, getTestLoader, getLoader
from Constant import Constants as C
import torch.optim as optim
from evaluation import eval

print('Dataset: ' + C.DATASET + ', Learning Rate: ' + str(C.LR) + '\n')

# Inicializar modelo DKT con constantes
model = DKT(C.INPUT, C.HIDDEN, C.LAYERS, C.OUTPUT)

# Inicializar optimizador Adam con tasa de aprendizaje
optimizer_adam = optim.Adam(model.parameters(), lr=C.LR)

# Inicializar optimizador Adagrad con tasa de aprendizaje
optimizer_adgd = optim.Adagrad(model.parameters(),lr=C.LR)

# Inicializar función de pérdida
loss_func = eval.lossFunc()

# Obtener DataLoader de entrenamiento y prueba
trainLoaders, testLoaders = getLoader(C.DATASET)

# Iterar sobre las épocas
for epoch in range(C.EPOCH):
    print('epoch: ' + str(epoch))
    model, optimizer = eval.train(trainLoaders, model, optimizer_adgd, loss_func)
    eval.test(testLoaders, model) 
 
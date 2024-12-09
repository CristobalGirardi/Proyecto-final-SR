import numpy as np
from torch.utils.data.dataset import Dataset
from Constant import Constants as C
import torch

class DKTDataSet(Dataset):
    def __init__(self, ques, ans):
        self.ques = ques
        self.ans = ans

    def __len__(self): 
        return len(self.ques)
    
    #Obtiene un elemento específico y lo convierte a one-hot encoding.
    def __getitem__(self, index):
        # Obtener preguntas y respuestas para el índice específico
        questions = self.ques[index]
        answers = self.ans[index]
        # Convertir a one-hot encoding
        onehot = self.onehot(questions, answers)
        # Convertir a tensor float de PyTorch y convertir lista a tensor
        return torch.FloatTensor(onehot.tolist())


    def onehot(self, questions, answers):

        # Inicializar matriz de ceros de tamaño máximo
        result = np.zeros(shape=[C.MAX_STEP, 2 * C.NUM_OF_QUESTIONS])
        # Iterar sobre los pasos máximos
        for i in range(C.MAX_STEP):
            # Si la respuesta es positiva, marcar como 1 en la pregunta correspondiente
            if answers[i] > 0:
                result[i][questions[i]] = 1
            # Si la respuesta es negativa, marcar como 1 en la pregunta correspondiente al final
            elif answers[i] == 0:
                result[i][questions[i] + C.NUM_OF_QUESTIONS - 1] = 1
        # Devolver la matriz one-hot
        return result 
import numpy as np
from data.DKTDataSet import DKTDataSet
import itertools
import tqdm

class DataReader():
    def __init__(self, path, maxstep, numofques):
        self.path = path
        self.maxstep = maxstep
        self.numofques = numofques
    
    # Función principal para obtener los datos de entrenamiento
    def getTrainData(self):
        trainqus = []
        trainans = []
        with open(self.path, 'r') as train:
            for lenght, ques, ans in tqdm.tqdm(itertools.zip_longest(*[train] * 3), desc='loading train data:    ', mininterval=2):
                # Procesar cada línea del archivo (preguntas y respuestas)
                lenght = int(lenght.strip().strip(','))
                ques = np.array(ques.strip().strip(',').split(',')).astype(int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(int)

                # Ajustar el número de preguntas y respuestas al paso máximo
                mod = 0 if lenght%self.maxstep == 0 else (self.maxstep - lenght%self.maxstep)
                zero = np.zeros(mod) - 1
                ques = np.append(ques, zero)
                ans = np.append(ans, zero)
                trainqus.append(ques)
                trainans.append(ans)
            
            # Concatenar todas las preguntas y respuestas en un array
            trainqus = np.concatenate(trainqus).astype(int)
            trainans = np.concatenate(trainans).astype(int)
            
        return trainqus.reshape([-1, self.maxstep]), trainans.reshape([-1, self.maxstep])

    # Función principal para obtener los datos de prueba
    def getTestData(self):
        testqus = []
        testans = []
        with open(self.path, 'r') as test:
            for lenght, ques, ans in tqdm.tqdm(itertools.zip_longest(*[test] * 3), desc='loading test data:    ', mininterval=2):
                
                # Procesar cada línea del archivo (preguntas y respuestas)
                lenght = int(lenght.strip().strip(','))
                ques = np.array(ques.strip().strip(',').split(',')).astype(int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(int)
                
                # Ajustar el número de preguntas y respuestas al paso máximo
                mod = 0 if lenght % self.maxstep == 0 else (self.maxstep - lenght % self.maxstep)
                zero = np.zeros(mod) - 1
                ques = np.append(ques, zero)
                ans = np.append(ans, zero)
                testqus.append(ques)
                testans.append(ans)
                
            #  Concatenar todas las preguntas y respuestas en un array
            testqus = np.concatenate(testqus).astype(int)
            testans = np.concatenate(testans).astype(int)

        return testqus.reshape([-1, self.maxstep]), testans.reshape([-1, self.maxstep])  
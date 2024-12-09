import pandas as pd
import ast
import random

# Esta función toma una lista de strings que representan arrays de preguntas y respuestas y los junta en un solo array
def juntar_qa_arrays(qa_arrays):
    ejercicios = []
    resultados = []
    for qa_array in qa_arrays:
        parsed_array = eval(qa_array)
        ejercicios.extend(parsed_array[0]) 
        resultados.extend(parsed_array[1])   
    combined_qa_array = [ejercicios, resultados]
    return str(combined_qa_array)  

# Leer el dataset de recordDS
recordDS = pd.read_csv("recordDS.csv")
recordDS_agrupado = recordDS.groupby('user_id').agg({
    'opern_id': 'first',             
    'record_id': 'first',           
    'create_time': 'first',       
    'qa_array': juntar_qa_arrays
}).reset_index()

# Iterar sobre cada fila del dataset
for index, row in recordDS.iterrows():
    
    # Convertir el string de la columna qa_array a una lista
    qa_list = ast.literal_eval(row["qa_array"])
    qa = ','.join(map(str, qa_list[0] ))
    ra = ','.join(map(str, qa_list[1] ))

    # Escribir en el archivo train o test dependiendo de un número aleatorio para separar los datos en train y test
    if random.random() < 0.9: #Separar train con test
        with open('../recordDS/train.csv', 'a') as train_file:
            train_file.write(f"{len(qa_list[0])}\n")
            train_file.write(f"{qa},\n")
            train_file.write(f"{ra}\n")
    else:
        with open('../recordDS/test.csv', 'a') as test_file:
            test_file.write(f"{len(qa_list[0])}\n")
            test_file.write(f"{qa},\n")
            test_file.write(f"{ra}\n") 

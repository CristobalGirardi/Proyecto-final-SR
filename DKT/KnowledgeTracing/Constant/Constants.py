Dpath = '../../KTDataset'

datasets = {
    'recordDS' : 'recordDS'
}

# question number of each dataset
numbers = {
    'recordDS' : 2586
}

DATASET = datasets['recordDS']
NUM_OF_QUESTIONS = numbers['recordDS']

# the max step of RNN model
MAX_STEP = 50
BATCH_SIZE = 64
LR = 0.002
EPOCH = 4
#input dimension
INPUT = NUM_OF_QUESTIONS * 2
# embedding dimension
EMBED = NUM_OF_QUESTIONS
# hidden layer dimension
HIDDEN = 200
# nums of hidden layers
LAYERS = 5
# output dimension
OUTPUT = NUM_OF_QUESTIONS
 
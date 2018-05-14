import numpy as np

a = np.array([1,2,3])
print(a)


'''
Kears not suitable for this task,
still use tensorflow

import numpy as np
from keras.layers import Input, Dense, Average, Add
from keras.models import Model

def searched_cell(weights_path=None):
    x = Input(shape=(20,))
    h = Input(shape=(1000,))
    node0x = Dense(1000, name='node0x')(x)
    node0h = Dense(1000, name='node0h')(h)
    node0 = Add()([node0x, node0h])
    node1 = Dense(1000, activation='relu', name='node1')(node0)
    node2 = Dense(1000, activation='tanh', name='node2')(node0)
    node3 = Dense(1000, activation='sigmoid', name='node3')(node0)
    node4 = Dense(1000, name='node4')(node0)
    node5 = Dense(1000, activation='tanh', name='node5')(node5)
    node6 = Dense(1000, activation='sigmoid', name='node6')(node2)
    node7 = Dense(1000, name='node7')(node0)
    node8 = Dense(1000, name='node8')(node2)
    node9 = Dense(1000, activation='relu', name='node9')(node3)
    node10 = Dense(1000, activation='relu', name='node10')(node6)
    node11 = Dense(1000, activation='sigmoid', name='node11')(node1)
    output = Average()([node4, node5, node7, node8, node9, node10, node11])

    searched_cell = Model(input=[x, h], output=output)

    if weight_path:
        searched_cell.load_weights(weights_path)

    return searched_cell

class searched_rnn():
'''
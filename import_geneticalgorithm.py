from built_modules import import_ann as ann
from copy import deepcopy
import numpy as np, random

def rep(a: ann.NeuralNetwork, b: ann.NeuralNetwork, average: bool = False, normalised_weights: tuple[float, float] = (0.5, 0.5)) -> ann.NeuralNetwork:
    new_org = ann.NeuralNetwork(a.all_neurons)
    if average:
        for i in range(a.n - 1):
            new_org.weights[i] = (normalised_weights[0] * np.array(a.weights[i]) + normalised_weights[1] * np.array(b.weights[i])) / 2
            new_org.biases[i] = (normalised_weights[0] * np.array(a.biases[i]) + normalised_weights[1] * np.array(b.biases[i])) / 2
            # for j in range(len(a.weights[i])):
            #     for k in range(len(a.weights[i][j])):
            #         new_org.weights[i][j][k] = (a.weights[i][j][k] + b.weights[i][j][k]) / 2
        # for i in range(len(a.biases)):
        #     for j in range(len(a.biases[i])):
        #         new_org.biases[i][j] = (a.biases[i][j] + b.biases[i][j]) / 2
    else:
        for i in range(a.n - 1):
            rand_arr_1 = np.random.randint(2, size = np.array(a.weights[i]).shape)
            rand_arr_2 = 1 - rand_arr_1
            new_org.weights[i] = rand_arr_1 * a.weights[i] + rand_arr_2 * b.weights[i]
            rand_arr_1 = np.random.randint(2, size = np.array(a.biases[i]).shape)
            rand_arr_2 = 1 - rand_arr_1
            new_org.biases[i] = rand_arr_1 * a.biases[i] + rand_arr_2 * b.biases[i]
        # for i in range(len(a.weights)):
        #     for j in range(len(a.weights[i])):
        #         for k in range(len(a.weights[i][j])):
        #             if random.choice([0, 1]):
        #                 new_org.weights[i][j][k] = a.weights[i][j][k]
        #             else:
        #                 new_org.weights[i][j][k] = b.weights[i][j][k]
        # for i in range(len(a.biases)):
        #     for j in range(len(a.biases[i])):
        #         if random.choice([0, 1]):
        #             new_org.biases[i][j] = a.biases[i][j]
        #         else:
        #             new_org.biases[i][j] = b.biases[i][j]
    return new_org

def mutate(a: ann.NeuralNetwork, r: float) -> ann.NeuralNetwork:
    mutated_org = ann.NeuralNetwork(a.all_neurons)
    for i in range(len(a.weights)):
        for j in range(len(a.weights[i])):
            for k in range(len(a.weights[i][j])):
                c = random.randrange(0, 10 ** 10) / 10 ** 10
                if c < r:
                    mutated_org.weights[i][j][k] = random.randint(-100, 100) / 100
                else:
                    mutated_org.weights[i][j][k] = a.weights[i][j][k]

    for i in range(len(a.biases)):
        for j in range(len(a.biases[i])):
            c = random.randrange(0, 10 ** 10) / 10 ** 10
            if c < r:
                mutated_org.biases[i][j] = random.randint(-100, 100) / 100
            else:
                mutated_org.biases[i][j] = a.biases[i][j]

    return mutated_org

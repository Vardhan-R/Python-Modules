# If this is used with import_geneticalgorithm, then the weights and biases may be in numpy arrays instead of lists.
import math, pygame, random, shelve

class NeuralNetwork:
    def __init__(self, n: tuple[int] | list[int]): # has [number of neurons in the layer]
        self.n = len(n)
        self.all_neurons = n
        self.n_i_neurons = n[0]
        self.n_h_layers = self.n - 2
        self.n_h_neurons = n[1: -1]
        self.n_o_neurons = n[-1]
        self.weights = [] # has [layer_num][neuron_num][prev_layer_neuron_num]
        self.biases = [] # has [layer_num][neuron_num]
        self.z_lst = []
        self.activations = [] # of all the layers
        self.costs = []
        self.change_weights = []
        self.change_biases = []

        for i in range(self.n):
            if i != 0:
                temp_lst_1 = []
                temp_lst_2 = []
                for j in range(n[i]):
                    temp_lst_1.append([random.randint(-100, 100) / 100 for k in range(n[i - 1])])
                    temp_lst_2.append([0 for k in range(n[i - 1])])
                self.weights.append(temp_lst_1.copy())
                self.change_weights.append(temp_lst_2.copy())
                temp_lst_1.clear()
                temp_lst_2.clear()

                self.biases.append([random.randint(-100, 100) / 100 for j in range(n[i])])
                self.change_biases.append([0 for j in range(n[i])])
                self.z_lst.append([0 for j in range(n[i])])
            self.activations.append([0 for j in range(n[i])])
            if i == self.n - 1:
                self.costs = [0 for j in range(n[i])]

    def show(self, screen: pygame.display, canvas: tuple[int, int] | list[int, int]):
        for i in range(self.n):
            for j in range(self.all_neurons[i]):
                temp_coor_1 = (canvas[0] * (i + 1 / 2) / self.n, canvas[1] * (j + 1 / 2) / self.all_neurons[i])
                if i != 0:
                    for k in range(self.all_neurons[i - 1]):
                        temp_coor_2 = (canvas[0] * (i - 1 / 2) / self.n, canvas[1] * (k + 1 / 2) / self.all_neurons[i - 1])
                        temp_clr = 255 * (2 * sigmoid(abs(self.weights[i - 1][j][k])) - 1)
                        if self.weights[i - 1][j][k] < 0:
                            pygame.draw.line(screen, (temp_clr, 0, 0), temp_coor_1, temp_coor_2, 2)
                        else:
                            pygame.draw.line(screen, (0, temp_clr, 0), temp_coor_1, temp_coor_2, 2)
        for i in range(self.n):
            for j in range(self.all_neurons[i]):
                shade = 255 * self.activations[i][j]
                pygame.draw.circle(screen, (shade, shade, shade), (canvas[0] * (i + 1 / 2) / self.n, canvas[1] * (j + 1 / 2) / self.all_neurons[i]), 20)

        for i in range(1, self.n):
            for j in range(self.all_neurons[i]):
                temp_clr = 255 * (2 * sigmoid(abs(self.biases[i - 1][j])) - 1)
                if self.biases[i - 1][j] < 0:
                    pygame.draw.circle(screen, (temp_clr, 0, temp_clr), (canvas[0] * (i + 1 / 2) / self.n, canvas[1] * (j + 1 / 2) / self.all_neurons[i] - 32), 10)
                else:
                    pygame.draw.circle(screen, (temp_clr, temp_clr, 0), (canvas[0] * (i + 1 / 2) / self.n, canvas[1] * (j + 1 / 2) / self.all_neurons[i] - 32), 10)

    def feedForward(self, input_lst: list[float | int]):
        for i in range(self.all_neurons[0]):
            self.activations[0][i] = sigmoid(input_lst[i])

        for i in range(1, self.n):
            for j in range(self.all_neurons[i]):
                temp = 0
                for k in range(self.all_neurons[i - 1]):
                    temp += self.activations[i - 1][k] * self.weights[i - 1][j][k] + self.biases[i - 1][j]
                self.z_lst[i - 1][j] = temp
                self.activations[i][j] = sigmoid(self.z_lst[i - 1][j])

    def calcCosts(self, ans: list[float | int]):
        for i in range(self.all_neurons[-1]):
            self.costs[i] = (ans[i] - self.activations[-1][i]) ** 2

    def backProp(self, ans: list[float | int], learning_rate: float | int):
        for i in range(1, self.n): # layer num
            for j in range(self.all_neurons[i]): # "i" layer's neuron num
                for k in range(self.all_neurons[i - 1]): # "i - 1" layer's neuron num
                    self.change_weights[i - 1][j][k] = self.pD_C_w(i, j, k, ans)
                self.change_biases[i - 1][j] = self.pD_C_b(i, j, ans)

        for i in range(1, self.n): # layer num
            for j in range(self.all_neurons[i]): # "i" layer's neuron num
                for k in range(self.all_neurons[i - 1]): # "i - 1" layer's neuron num
                    self.weights[i - 1][j][k] -= learning_rate * self.change_weights[i - 1][j][k]
                self.biases[i - 1][j] -= learning_rate * self.change_biases[i - 1][j]

    def pD_C_a(self, m: int, k: int, ans: list[float | int]) -> float: # pD_C_a ==> partial differential of cost with respect to activation
        if m < self.n - 1:
            temp = 0
            for i in range(self.all_neurons[m + 1]):
                temp += self.pD_C_a(m + 1, i, ans) * sigmoid(self.z_lst[m][i]) * sigmoid(-self.z_lst[m][i]) * self.weights[m][i][k]
            return temp
        else:
            return 2 * (self.activations[-1][k] - ans[k])

    def pD_C_w(self, m: int, k: int, j: int, ans: list[float | int]) -> float: # pD_C_w ==> partial differential of cost with respect to weight
        return self.pD_C_a(m, k, ans) * sigmoid(self.z_lst[m - 1][k]) * sigmoid(-self.z_lst[m - 1][k]) * self.activations[m - 1][j]

    def pD_C_b(self, m: int, k: int, ans: list[float | int]) -> float: # pD_C_b ==> partial differential of cost with respect to bias
        return self.pD_C_a(m, k, ans) * sigmoid(self.z_lst[m - 1][k]) * sigmoid(-self.z_lst[m - 1][k])

    def best(self) -> int:
        temp = 0
        for i in range(self.all_neurons[-1]):
            if self.activations[-1][i] > self.activations[-1][temp]:
                temp = i
        return temp

    def saveNeuralNetwork(self, file_path_and_name: str):
        d = shelve.open(file_path_and_name, writeback = True)
        d["weights"] = self.weights
        d["biases"] = self.biases
        d.close()

    def loadNeuralNetwork(self, file_path_and_name: str):
        d = shelve.open(file_path_and_name)
        self.weights = d["weights"]
        self.biases = d["biases"]
        d.close()

def sigmoid(x: float | int) -> float | int:
    try:
        return 1 / (1 + math.e ** (-x))
    except:
        return 0

def relu(x: float | int) -> float:
    return (x + abs(x)) / 2

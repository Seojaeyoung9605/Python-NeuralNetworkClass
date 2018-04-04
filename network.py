from random import *
from math import *
from time import time


class Neuron:
    def __init__(self, value, comp, pos, error, inc, outc):
        self.value = value   # The value of the neuron
        self.comp = comp     # Neuron's component
        self.pos = pos       # Neuron's position in the component
        self.error = error   # Neuron's error
        self.inc = inc[::]   # Indexes of INCOMING neurons  (comp-1 component)
        self.outc = outc[::] # Indexes of OUTCOMING neurons (comp+1 component)
    
    def __str__(self): 
        return ("Neuron(" + str(self.value) + ", " + str(self.comp)  + ", " + 
                            str(self.pos)   + ", " + str(self.error) + ", " + 
                            str(self.inc)   + ", " + str(self.outc)    + ")")


class NeuralNetwork:
    def __init__(self, template=[0], syn_prc=100, co=0.7, nmin=0, nmax=1):
        self.template = template  # Template for the net (template[i] = 
                                  #         number of neurons in 'i' component)

        self.syn_prc = syn_prc    # The percent of synapses, that the network 
                                  # will create. By default each neuron is 
                                  # connected to it's every neighbour.

        self.co = co              # Almost a bias.

        self.nmin = nmin          # This is the minimum possible value that 
                                  # can be received from the input.

        self.nmax = nmax          # And the largest possible value. 
                                  # We need this because all the neurons' 
                                  # impulses must be in the range(0, 1), so we 
                                  # have to map input and output to that range.
        
        # Creating the matrix of edges between neurons 
        # matrix[x][i][j] is the edge 
        # from the neuron on pos 'i' in comp 'x'
        #   to the neuron on pos 'j' in comp 'x + 1'
        self.matrix = ([
            [
                [0 for to in range(template[comp + 1])] 
                for frm in range(template[comp])
            ] 
            for comp in range(len(template) - 1)
        ])

        # Creating a list of neurons with random values
        self.neurons = ([
            [
                Neuron(randint(-100, 100) / 100, comp, pos, -1, [], []) 
                for pos in range(template[comp])
            ] 
            for comp in range(len(template))
        ])
        
        # Adding outcoming neurons
        for comp in range(len(template) - 1):  # Cycling through components
            for n1 in range(template[comp]):   # The 'main' neuron
                for n2 in range(template[comp + 1]):  # Its neighbours
                    if randint(0, 99) < syn_prc:
                        self.neurons[comp][n1].outc.append(n2)
                        self.neurons[comp + 1][n2].inc.append(n1)
                        self.matrix[comp][n1][n2] = randint(-100, 100) / 100

                if len(self.neurons[comp][n1].outc) == 0:     # We have to keep
                    n2 = randint(0, template[comp + 1] - 1)   # the network 
                    self.neurons[comp][n1].outc.append(n2)    # connected.
                    self.neurons[comp + 1][n2].inc.append(n1)
                    self.matrix[comp][n1][n2] = randint(-100, 100) / 100

        # Adding incoming neurons, if we didn't add some
        # (every neuron has to have incoming edges)
        for comp in range(1, len(template)):
            for n1 in range(template[comp]):
                if len(self.neurons[comp][n1].inc) == 0:
                    n2 = randint(0, template[comp - 1] - 1)
                    self.neurons[comp][n1].inc.append(n2)
                    self.neurons[comp - 1][n2].outc.append(n1)
                    self.matrix[comp - 1][n2][n1] = randint(-100, 100) / 100
        
    
    def map_to(self, n) :   # This function maps 'n' to the range(0, 1)
        return n / (self.nmax - self.nmin)
    
    def map_from(self, n):  # And this function maps its value back
        return n * (self.nmax - self.nmin)
    
    # Returns weighted sum of the input edges
    def weighted_sum(self, n):
        sm = 0
        for pos in n.inc:  # Cycling through incoming synapses
            sm += (self.matrix[n.comp - 1][pos][n.pos] * 
                   self.neurons[n.comp - 1][pos].value)
        return sm
    
    # Sigma function
    def sig(self, n):
        return (1 / (1 + exp(-(self.weighted_sum(n)))))
    
    # Derivative for that sigma function
    def der(self, n):
        return (self.sig(n)) * (1 - self.sig(n))
    
    # Recounting error for the neuron 'n'
    def recount_error(self, n):
        new_error = 0
        for pos in n.outc:
            new_error += (self.neurons[n.comp + 1][pos].error * 
                          self.matrix[n.comp][n.pos][pos])
        n.error = new_error
        return new_error
    
    # Recounting all input edges for the neuron 'n'
    def recount_edges(self, n):
        for pos in n.inc:
            self.matrix[n.comp - 1][pos][n.pos] += (self.co * n.error * 
                    self.der(n) * self.neurons[n.comp - 1][pos].value)
    
    def train(self, filename="train.txt", verbose=False, rep=1):
        tasks = open(filename).readlines()
        
        if verbose:
            allcount = len(tasks) * rep
            cnt = 0
            prcnt = 0
            stime = time()
        
        for rp in range(rep):            
            for task in tasks:
                if verbose:   # Displaying the job, that is already done
                    cnt += 1  # and the time that is left
                    if int(cnt / allcount * 100) > prcnt:
                        prcnt += 1
                        timeleft = get_time(int((100 - prcnt) * 
                                           ((time() - stime) / prcnt)))

                        print('{}%; Time left: {}'.format(str(prcnt).zfill(2),
                                                                     timeleft))
                        
                # Input and output values are all in one line. 
                # Input values are the first template[0] ones, 
                # and output - the rest template[-1].
                values = list(map(int, task.split()))  
                
                # Dividing the values according to the template    
                inp, out = [], []
                for i in range(self.template[0] + self.template[-1]):
                    if i >= self.template[0]: 
                        out.append(self.map_to(values[i]))
                    else: 
                        inp.append(self.map_to(values[i]))
                
                # Changing input neuron values
                for pos in range(self.template[0]):
                    self.neurons[0][pos].value = inp[pos]
                
                # Counting all other neurons' values
                for comp in range(1, len(self.neurons)):
                    for pos in range(len(self.neurons[comp])):
                        val = self.sig(self.neurons[comp][pos])
                        self.neurons[comp][pos].value = val
                        
                # Counting errors for the last neurons
                for pos in range(self.template[-1]):        
                    err = out[pos] - self.neurons[-1][pos].value
                    self.neurons[-1][pos].error = err
                
                # Recounting errors
                for comp in range(len(self.neurons) - 2, -1, -1):
                    for pos in range(len(self.neurons[comp])):
                        self.recount_error(self.neurons[comp][pos])
                
                # Recounting edges
                for comp in range(1, len(self.neurons)):
                    for pos in range(len(self.neurons[comp])):
                        self.recount_edges(self.neurons[comp][pos])  

        if verbose:
            print('Done. Time used: ' + get_time(int(time() - stime)))
       
    def predict(self, inp):
        # Changing input neuron values
        for pos in range(self.template[0]):
            self.neurons[0][pos].value = self.map_to(inp[pos])
        
        # Counting all other neurons' values
        for comp in range(1, len(self.neurons)):
            for pos in range(len(self.neurons[comp])):
                val = self.sig(self.neurons[comp][pos])
                self.neurons[comp][pos].value = val
        
        # And returning the array of all output neurons
        out = []
        for pos in range(self.template[-1]):
            out.append(self.map_from(self.neurons[-1][pos].value))

        # Returning a copy just in case
        return out[::]
    
    def save(self, filename="network.net"):
        fout = open(filename, 'w')
        print(self.template, file=fout)
        print(self.co,       file=fout)
        print(self.syn_prc,  file=fout)
        print(self.nmin,     file=fout)
        print(self.nmax,     file=fout)
        print(self.matrix,   file=fout)

        compstrings = [list(map(str, comp)) for comp in (self.neurons)]
        print(str(compstrings).replace("'", ''), file=fout)
        fout.close()
    
    def load(self, filename="network.net"):
        (template, co, syn_prc,
                nmin, nmax, matrix, neurons) = open(filename).readlines()
        self.template = eval(template)
        self.co =       eval(co)
        self.syn_prc =  eval(syn_prc)
        self.nmin =     eval(nmin)
        self.nmax =     eval(nmax)
        self.matrix =   eval(matrix)
        self.neurons =  eval(neurons)
    
    def print(self):
        nmax = max(self.template) * 2 - 1
        net = []
        for i in range(len(self.template)):
            line = (' ' * ((nmax - (self.template[i] * 2 - 1)) // 2) + 
                    ' '.join(["0" for j in range(self.template[i])]) + 
                    ' ' * ((nmax - (self.template[i] * 2 - 1)) // 2))
            net.append(line)
        
        print("\nNeurons: ")
        for n in self.neurons: print(*list(map(str, n)))
        print("\nSynapses ")
        for i in self.matrix: print(*i)
        print("\nModel: ")
        for pos in range(nmax):
            for comp in range(len(net)):
                print(net[comp][pos], end='  ')
            print()


def get_time(seconds_all):
    seconds = str(seconds_all % 60).zfill(2)
    minutes = str((seconds_all // 60) % 60).zfill(2)
    hours = str((seconds_all // 3600)).zfill(2)
    return hours + 'h:' + minutes + 'm:' + seconds + 's'


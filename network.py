from random import *
from math import *
from time import time


class Neuron:
    def __init__(self, value, comp, pos, mistake, inc, outc):
        self.value = value  # The value of the neuron
        self.comp = comp  # Neuron's component
        self.pos = pos  # Neuron's position in the component
        self.mistake = mistake  # Neuron's mistake
        self.inc = inc[::]  # Indexes of INCOMING neurons (they are all in comp-1 component)
        self.outc = outc[::]  # Indexes of OUTCOMING neurons (they are all in comp+1 component)
    
    def __str__(self):
        return "Neuron(" + str(self.value) + ", " + str(self.comp) + ", " + str(self.pos) + ", " + str(self.mistake) + ", " + str(self.inc) + ", " + str(self.outc) + ")" 


class NeuralNetwork:
    def __init__(self, template=[0], syn_prc=100, co=0.7, nmin=0, nmax=1):
        self.template = template  # Template for the net (template[i] = number of neurons in 'i' component)
        self.syn_prc = syn_prc  # The percent of synapses, that the network will create. By default each neuron is connected with each.
        self.co = co  # Idk.
        self.nmin = nmin  # This is the minimum possible value, that can be received from the input
        self.nmax = nmax  # And the largest possible value. We need this because all the neurons' impulses must be in the range(0, 1), so we are going to 'transform' input and output to that range.
        
        #Creating the matrix of edges between neurons (matrix[x][i][j] is the edge from the neurons on pos 'i' in comp 'x' to the neuron on pos 'j' in comp 'x + 1'
        self.matrix = [[[0 for to in range(template[comp + 1])] for frm in range(template[comp])] for comp in range(len(template) - 1)]        
        #Creating neurons array
        self.neurons = [[Neuron(randint(-100, 100) / 100, comp, pos, -1, [], []) for pos in range(template[comp])] for comp in range(len(template))]
        
        #Adding outcoming neurons
        for comp in range(len(template) - 1):  # Cycling for components
            for n1 in range(template[comp]):  # The 'main' neuron
                for n2 in range(template[comp + 1]):  # Main neuron's neighbours
                    if randint(0, 99) < syn_prc:
                        self.neurons[comp][n1].outc.append(n2)
                        self.neurons[comp + 1][n2].inc.append(n1)
                        self.matrix[comp][n1][n2] = randint(-100, 100) / 100
                if len(self.neurons[comp][n1].outc) == 0:  # If we didn't add anything
                    n2 = randint(0, template[comp + 1] - 1)
                    self.neurons[comp][n1].outc.append(n2)
                    self.neurons[comp + 1][n2].inc.append(n1)
                    self.matrix[comp][n1][n2] = randint(-100, 100) / 100
        #Adding incoming neurons, if we didn't add some
        for comp in range(1, len(template)):
            for n1 in range(template[comp]):
                if len(self.neurons[comp][n1].inc) == 0:
                    n2 = randint(0, template[comp - 1] - 1)
                    self.neurons[comp][n1].inc.append(n2)
                    self.neurons[comp - 1][n2].outc.append(n1)
                    self.matrix[comp - 1][n2][n1] = randint(-100, 100) / 100
        
    
    def toRange(self, n):  # This function puts 'n' into the range(0, 1)
        #return (n - (self.nmax - self.nmin) // 2) / ((self.nmax - self.nmin) // 2 + 1)
        return n / (self.nmax - self.nmin)
    
    def fromRange(self, n):  # And this function gets the value back
        #return n * ((self.nmax - self.nmin) // 2 + 1) + (self.nmax - self.nmin) // 2
        return n * (self.nmax - self.nmin)
    
    #Returning the weighted (if w) sum of the input edges
    def sum(self, n, w=1):
        sm = 0
        for pos in range(len(self.neurons[n.comp - 1])):
            if w: sm += self.matrix[n.comp - 1][pos][n.pos] * self.neurons[n.comp - 1][pos].value
            else: sm += self.matrix[n.comp - 1][pos][n.pos]
        return sm
    
    #Sigma function
    def sig(self, n):
        return ( 1 / (1 + exp(-(self.sum(n)))) )
    
    #Derivative for that sigma function
    def der(self, n):
        return (self.sig(n)) * (1 - self.sig(n))
    
    #Recounting mistake for the neuron 'n'
    def recount_mistake(self, n):
        new_mist = 0
        for pos in range(len(self.neurons[n.comp + 1])):
            new_mist += self.neurons[n.comp + 1][pos].mistake * self.matrix[n.comp][n.pos][pos]
        n.mistake = new_mist
        return new_mist
    
    #Recounting all input edges for the neuron 'n'
    def recount_edges(self, n):
        for pos in range(len(self.neurons[n.comp - 1])):
            self.matrix[n.comp - 1][pos][n.pos] += self.co * n.mistake * self.der(n) * self.neurons[n.comp - 1][pos].value      
    
    def educate(self, filename="education.txt", show_process=False, rep=1):
        tasks = open(filename).readlines()
        
        if show_process:
            allcount = len(tasks) * rep
            cnt = 0
            prcnt = 0
            stime = time()
        
        for rp in range(rep):            
            for task in tasks:
                if show_process:  # Displaying the job, that is already done, and the time that is left
                    cnt += 1
                    if int(cnt / allcount * 100) > prcnt:
                        prcnt += 1
                        timeleft = getTime(int((100 - prcnt) * ((time() - stime) / prcnt)))
                        print(str(prcnt) + '%;  Time left: ' + timeleft)                
                variables = list(map(int, task.split()))  # Input and output variables are all in one line. Input variables are the first template[0] ones, and output = the rest (actually, it is template[-1])
                
                #Sorting the inputs according to the template    
                inp, out = [], []
                for i in range(self.template[0] + self.template[-1]):
                    if i >= self.template[0]: out.append(self.toRange(variables[i]))
                    else: inp.append(self.toRange(variables[i]))
                
                #Changing start values
                for pos in range(self.template[0]):
                    self.neurons[0][pos].value = inp[pos]
                
                #Counting values
                for comp in range(1, len(self.neurons)):
                    for pos in range(len(self.neurons[comp])):
                        self.neurons[comp][pos].value = self.sig(self.neurons[comp][pos])
                        
                #Counting mistakes for the last neurons
                for pos in range(self.template[-1]):        
                    self.neurons[-1][pos].mistake = out[pos] - self.neurons[-1][pos].value
                
                #Recounting mistakes
                for comp in range(len(self.neurons) - 2, -1, -1):
                    for pos in range(len(self.neurons[comp])):
                        self.recount_mistake(self.neurons[comp][pos])
                
                #Recounting edges
                for comp in range(1, len(self.neurons)):
                    for pos in range(len(self.neurons[comp])):
                        self.recount_edges(self.neurons[comp][pos])  
        if show_process: print('Done. Time used: ' + getTime(int(time() - stime)))        
       
    def check(self, inp):
        #Changing start values
        for pos in range(self.template[0]):
            self.neurons[0][pos].value = self.toRange(inp[pos])
        
        #Counting values
        for comp in range(1, len(self.neurons)):
            for pos in range(len(self.neurons[comp])):
                self.neurons[comp][pos].value = self.sig(self.neurons[comp][pos])  
        
        #And returning the array of all answer neurons
        ans = []
        for pos in range(self.template[-1]):
            ans.append(self.fromRange(self.neurons[-1][pos].value))
        return ans
    
    def save(self, filename="network.txt"):
        fout = open(filename, 'w')
        print(self.template, file=fout)
        print(self.co, file=fout)
        print(self.syn_prc, file=fout)
        print(self.nmin, file=fout)
        print(self.nmax, file=fout)
        print(self.matrix, file=fout)
        print(str([list(map(str, comp)) for comp in (self.neurons)]).replace("'", ''), file=fout)
        fout.close()
    
    def load(self, filename="network.txt"):
        template, co, syn_prc, nmin, nmax, matrix, neurons = open(filename).readlines()
        self.template = eval(template)
        self.co = eval(co)
        self.syn_prc = eval(syn_prc)
        self.nmin = eval(nmin)
        self.nmax = eval(nmax)
        self.matrix = eval(matrix)
        self.neurons = eval(neurons)


def getTime(seconds_all):
    seconds = str(seconds_all % 60)
    if len(seconds) == 1: seconds = '0' + seconds
    minutes = str((seconds_all // 60) % 60)
    if len(minutes) == 1: minutes = '0' + minutes
    hours = str((seconds_all // 3600))
    if len(hours) == 1: hours = '0' + hours
    ans = hours + 'h:' + minutes + 'm:' + seconds + 's'
    return ans
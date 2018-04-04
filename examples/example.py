import sys
sys.path.append("..")
import network

    
# In this example I am going to show you how to work with my neural network

# This line is creating the neural network. The constructor has 5 arguments:
#  - template  In this list you can select the amount of neurons in
#              each component.  Default - [0]
#  - syn_prc   By changing this parameter you can change the amount of synapses
#              in your network - from default 100% (each with each) to 1% 
#              (each neuron has one input and one output). So, if the value is 
#              small, the network works faster, but the error is bigger.  
#              Default - 100
#  - co        Learning coefficient. Some sort of a bias.  Default = 0.7
#  - nmin      The minimum value, that the network will get/give 
#              (positive only).  Default - 0
#   -nmax      The maximum value, that the network will get/give 
#              (positive only).  Default - 1
nw = network.NeuralNetwork(template=[1, 4, 9, 10, 1], 
                           syn_prc=90, 
                           co=0.7, 
                           nmin=0, 
                           nmax=3000)

# Now we are training the network. The 'train' method has 5 arguments:
#  - filename  This is the name of the file with training data
#              (has to be in the current directory).  Default - 'train.txt'
#  - verbose   If true, this will show you the training process
#              (percentage, time left, etc.).  Default - False
#  - rep       Repetitions. The number of times that the network will train on
#              the given file. More repetitions - better result.  Default - 1
#
# In the training file one line is one task. First template[0] numbers are the 
# input numbers, and the rest template[-1] numbers are the desired outputs.
nw.train(filename="train.txt", verbose=True, rep=10)

# After training you can print the network to see what's inside.
nw.print()

# So, now we've got the trained network. Let's make it predict things.
# In this example the network was trained to multiply numbers in range(1, 1000)
# by 3. The 'predict' method receives a list with length = template[0] of 
# values for the input neurons. They have to be in range(nmin, nmax).
# In this case, we have one input neuron, so the list is a single number.
a = int(input("Input value\n>>> "))
print(nw.predict([a]))

# Well, it doesn't predict perfectly, but it still quite close to real answers.
# Now, if you don't want to lose your trained network, you can save it. 
# The 'save' method accepts a filename as an argument and saves the network's
# data to it. The default filename is 'network.net'.
nw.save(filename="network.net")

# Once you have a saved network, you can load the it in your program using 
# the 'load' method, which also accepts a filename as an argument.
nw2 = network.NeuralNetwork()
nw2.load(filename="network.net")

# After the second network is created and loaded, it's ready to predict.
a = int(input("Input value\n>>> "))
print(nw2.predict([a]))

# If you want to experiment a bit more, there are a couple of already 
# trained networks in the 'trained_networks' folder.


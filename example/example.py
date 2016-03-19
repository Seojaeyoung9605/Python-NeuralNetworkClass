from network import *
    
#In this example I am going to show you how to work with my neuron network

#This line is creating the neuron network. There are 4 parameters, that you can change:
#-template. In this array you choose, how many neurons will be in each component.  Default - [0]
#-co. Learning coefficient. I don't know what this is used for :P  Default = 0.7
#-nmin. The minimum value, that the network will get/give (only positive).  Default - 0
#-nmax. The maximum value, that the network will get/give (only positive).  Default - 1
nw = NeuronNetwork(template=[1, 10, 10, 1], co=0.7, nmin=0, nmax=3000)

#Now we are educating the network. There are 2 parameters, that you can change:
#-filename. This is the name of the file (should be in the current directory).  Default - 'education.txt'
#-show_process. If true, this will show you the process of education (percentage, time left, etc.).  Default - False
#-rep. Repetitions. The number of times, that the network is going to educate on the given file.  Default - 1
#
#In the education file one line is one task. First template[0] numbers are the input numbers, and the rest template[-1] numbers are the output (desired answer) numbers.
nw.educate(filename="education.txt", show_process=True, rep=10)

#So, now we've got the educated network. Let's check something.
#In this example the network was educated to multiple numbers in range(1, 1000) by 3. Let's check.
#Function 'check' receives an array with length = template[0] of input signals for the input neurons. This values should be in range(nmin, nmax)
a = int(input())
print(nw.check([a]))

#The network is educated, and is working.. khm, quite good, yes.
#But the education took a lot of time. So now let's save it.
#In save function there is only one parameter - filename.  Default = 'network.txt'
nw.save(filename="network.txt")

#The network is saved, let's load it.
#Load function also has only one parameter, which is the same, as in the save function.
nw2 = NeuronNetwork()
nw2.load(filename="network.txt")

#NeuronNetwork2 is loaded, let's check it.
a = int(input())
print(nw2.check([a]))

#Working? Hope so.

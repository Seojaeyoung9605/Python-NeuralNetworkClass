# NeuralNetworkClass
This is a class in python for the neural network.  

## Creating the network
This line is creating the neural network.  
```python
nw = NeuralNetwork(template=[1, 10, 10, 1], syn_prc=80, co=0.7, nmin=0, nmax=3000)
```
The constructor has 5 arguments:  
- **template**. In this list you can select the amount of neurons in each component.  **Default - [0]**
- **syn_prc**. By changing this parameter you can change the amount of synapses in your network - from default 100% (each with each) to 1% (each neuron has one input and one output). So, if the value is small, the network works faster, but the error is bigger.  **Default - 100**
- **co**. Learning coefficient. Some sorf of a bias.  **Default = 0.7**
- **nmin**. The minimum value, that the network will get/give (positive only).  **Default - 0**
- **nmax**. The maximum value, that the network will get/give (positive only).  **Default - 1**

## Training the network
If you want your network to do something, you need to train it. You can do it with the ***train*** method:  
```python
nw.train(filename="train.txt", verbose=True, rep=10)
```
The ***train*** method has 3 arguments:  
- **filename**. This is the name of the file (has to be in the current directory).  **Default - 'train.txt'**
- **verbose**. If true, this will show you the training process (percentage, time left, etc.).  **Default - False**
- **rep**. Repetitions. The number of times that the network will train on the given file.  **Default - 1**

In the training file one line is one task. First *template[0]* numbers are the input numbers, and the rest *template[-1]* numbers are the desired outputs.  

## Using the network
So, now we've got the educated network. Let's make it predict things.
In this example the network was trained to multiply numbers in range(1, 1000) by 3.  
The ***check*** method receives a list with *length = template[0]* of values for the input neurons. They have to be in range(nmin, nmax).  
```python
values = list(map(int, input().split()))
print(nw.predict(values))
```

## Saving the network to file
If you don't want to lose your trained network, you can save it.  
The ***save*** method acceptc a filename as an argument and saves the network's data to it. **Default filename - 'network.net'**  
```python
nw.save(filename="network.net")
```

## Loading the network from file
Once you have a saved network, you can load it in your program using the ***load*** method, which also accepts a filename of the saved network as an argument. **Default filename - 'network.net'**  
```python
nw2 = NeuralNetwork()
nw2.load(filename="network.net")
```
Now, *nw2* is a copy of *nw1*, and it will work absolutely the same.  

> A working example of this program and a couple of already trained networks you can find in the *'examples'* folder.






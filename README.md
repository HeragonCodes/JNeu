# JNeu
JNEU is a simple light-weight java library for creating neural networks. By implementing ejml's optimized matrix library Jneu allows for fast and easy neural network setup, customization and testing. 

By utilizing the quick State.builStateToFile(int[] layerSizes, String filePath) function you can easily initialize a state with your options. As an example:
```
State.buildStateToFile(new int[]{240, 256, 128, 16}, "data/state");
```
Once initialized, you can load the state into a new NeuralNetwork object with the State.loadStateFromFile(String filePath), just like this:
```
NeuralNetwork network = State.loadStateFromFile("data/state");
```
And now you can have fun customizing your neural network as you wish. You could, for example:
- add a new layer to the neural network with ``` network.add(layer) ```
- remove a layer from the neural netwrok with ``` network.remove(layer) ```
- get a layer and change its activation type with ``` network.getLayer(0).setActivation(Layer.Act.LEAKY_RELU); ```
- add a custom activation function with ``` network.getLayer(1).setCustomActivation(x -> 2/(1+Math.exp(-2*x))); ```

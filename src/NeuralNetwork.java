import org.ejml.data.DMatrixRMaj;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork implements Serializable {

    @java.io.Serial
    private static final long serialVersionUID = 1L;

    private final List<Layer> allLayers;

    public NeuralNetwork(List<Layer> allLayers){
        this.allLayers = (allLayers != null) ? new ArrayList<>(allLayers) : new ArrayList<>();
    }
    public NeuralNetwork(){
        this.allLayers = new ArrayList<>();
    }

    public void add(Layer l){
        allLayers.add(l);
    }

    public void remove(Layer l){allLayers.remove(l);}

    public DMatrixRMaj forwardPass(DMatrixRMaj inputVector){
        if (allLayers.isEmpty()) throw new IllegalStateException("Cannot run forward pass: network has no layers.");
        if (inputVector == null) throw new IllegalArgumentException("Input vector cannot be null.");
        DMatrixRMaj vector = inputVector;
        for (Layer l : allLayers){
            l.setLastInput(vector);
            l.forwardPass();
            vector = l.getLastOutput();
        }
        return vector;
    }

    public Layer getLayer(int index){return allLayers.get(index);}
    public Layer getOutputLayer(){return allLayers.getLast();}
}

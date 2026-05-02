package com.github.heragoncodes.jneu;


import org.ejml.data.DMatrixRMaj;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork implements Serializable {

    @java.io.Serial
    private static final long serialVersionUID = 1L;

    private final List<Layer> allLayers;
    private LossFunctions.Loss lossFunction;

    public NeuralNetwork(List<Layer> allLayers, LossFunctions.Loss function){
        this.allLayers = (allLayers != null) ? new ArrayList<>(allLayers) : new ArrayList<>();
        this.lossFunction = function;
    }
    public NeuralNetwork(List<Layer> allLayers){this(allLayers, LossFunctions.Loss.MEAN_SQUARED);}
    public NeuralNetwork(){this.allLayers = new ArrayList<>();}

    public void add(Layer l){allLayers.add(l);}
    public void add(int index, Layer l){allLayers.add(index, l);}
    public void remove(Layer l){allLayers.remove(l);}
    public void remove(int index){allLayers.remove(index);}

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
    public void backPropagate(DMatrixRMaj expected){
        DMatrixRMaj inputError = null;
        switch (lossFunction){
            case MEAN_SQUARED -> inputError = LossFunctions.deriveMSE(getOutputLayer().getLastOutput(), expected);
            case MEAN_ABSOLUTE -> inputError = LossFunctions.deriveMAE(getOutputLayer().getLastOutput(), expected);
            case CROSS_ENTROPY -> inputError = LossFunctions.deriveCE(getOutputLayer().getLastOutput(), expected);
            case BINARY_CROSS_ENTROPY -> inputError = LossFunctions.deriveBCE(getOutputLayer().getLastOutput(), expected);
        }
        for (int i = allLayers.size()-1; i >= 0; i--) {
            inputError = getLayer(i).backwardPass(inputError);
        }
    }

    public void learn(DMatrixRMaj inputVector, DMatrixRMaj expected, boolean verboseLoss){
        DMatrixRMaj outputVector = forwardPass(inputVector);
        if (verboseLoss) System.out.println(calculateLoss(outputVector, expected));
        backPropagate(expected);
    }
    public void learn(DMatrixRMaj inputVector, DMatrixRMaj expected){
        learn(inputVector, expected, true);
    }

    public void optimize(double learningRate){
        for (Layer l : allLayers){
            l.updateLayer(learningRate);
        }
    }

    public double calculateLoss(DMatrixRMaj output, DMatrixRMaj expected){
        switch (lossFunction){
            case MEAN_SQUARED -> {return LossFunctions.applyMSE(output, expected);}
            case MEAN_ABSOLUTE -> {return LossFunctions.applyMAE(output, expected);}
            case CROSS_ENTROPY -> {return LossFunctions.applyCE(output, expected);}
            case BINARY_CROSS_ENTROPY -> {return LossFunctions.applyBCE(output, expected);}
        }
        throw new IllegalArgumentException("Specified loss function is inexistent");
    }

    public void setLossFunction(LossFunctions.Loss function){lossFunction = function;}

    public Layer getLayer(int index){return allLayers.get(index);}
    public Layer getOutputLayer(){return allLayers.getLast();}
}

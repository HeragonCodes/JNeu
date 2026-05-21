package com.github.heragoncodes.jneu;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.RandomMatrices_DDRM;

import java.io.*;
import java.util.Random;

public class State {

    public enum Init {
        ONE,
        ZERO,
        UNIFORM_BIPOLAR,
        HE_UNIFORM;
    }

    public static void buildStateToFile(int[] layerSizes, String filePath, Init initializationType, int seed) throws IOException {
        NeuralNetwork network = new NeuralNetwork();
        Random rand = new Random(seed);
        for (int i = 1; i < layerSizes.length; i++) {
            DMatrixRMaj weights = new DMatrixRMaj(layerSizes[i], layerSizes[i-1]);
            DMatrixRMaj biases = new DMatrixRMaj(layerSizes[i], 1);
            biases.zero();
            if (initializationType == Init.ZERO){
                weights.zero();
            } else if (initializationType == Init.ONE){
                weights.fill(1.0);
            } else {
                double limit = 0.0;
                switch (initializationType){
                    case UNIFORM_BIPOLAR -> limit = 1.0;
                    case HE_UNIFORM -> limit = Math.sqrt(6.0/layerSizes[i-1]);
                }
                RandomMatrices_DDRM.fillUniform(weights, -limit, limit, rand);
            }
            Layer l = new Layer(weights, biases);
            network.add(l);
        }
        saveStateToFile(network, filePath);
    }
    public static void buildStateToFile(int[] layerSizes, String filePath, Init initializationType) throws IOException {
        Random rand = new Random();
        buildStateToFile(layerSizes, filePath, initializationType, rand.nextInt());
    }
    public static void buildStateToFile(int[] layerSizes, String filePath) throws IOException {
        buildStateToFile(layerSizes, filePath, Init.HE_UNIFORM);
    }

    public static NeuralNetwork loadStateFromFile(String filePath) throws IOException, ClassNotFoundException{
        try (FileInputStream inStream = new FileInputStream(filePath);
             ObjectInputStream in = new ObjectInputStream(inStream)){
            return (NeuralNetwork) in.readObject();
        }
    }

    public static void saveStateToFile(NeuralNetwork network, String filePath) throws IOException {
        try (FileOutputStream outStream = new FileOutputStream(filePath);
             ObjectOutputStream out = new ObjectOutputStream(outStream)){
            out.writeObject(network);
        }
    }
}

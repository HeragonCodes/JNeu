package com.github.heragoncodes.jneu;


import org.ejml.data.DMatrixRMaj;

public class ActivationFunctions {
    public static void applyReLU(DMatrixRMaj vector){
        for (int i = 0; i < vector.getNumRows(); i++) {
            vector.data[i] = Math.max(0, vector.data[i]);
        }
    }
    public static void applyLeakyReLU(DMatrixRMaj vector){
        for (int i = 0; i < vector.getNumRows(); i++) {
            vector.data[i] = vector.data[i] > 0 ? vector.data[i] : 0.01 * vector.data[i];
        }
    }
    public static void applySigmoid(DMatrixRMaj vector){
        for (int i = 0; i < vector.getNumRows(); i++) {
            vector.data[i] = 1 / (1 + Math.exp(-vector.data[i]));
        }
    }
    public static void applyCustom(DMatrixRMaj vector, CustomFunction customFunction){
        for (int i = 0; i < vector.getNumRows(); i++) {
            vector.data[i] = customFunction.applyAsDouble(vector.data[i]);
        }
    }

    public static void deriveReLU(DMatrixRMaj vector){
        for (int i = 0; i < vector.getNumRows(); i++) {
            vector.data[i] = (vector.data[i] > 0) ? 1 : 0;
        }
    }
    public static void deriveLeakyReLU(DMatrixRMaj vector){
        for (int i = 0; i < vector.getNumRows(); i++) {
            vector.data[i] = (vector.data[i] > 0) ? 1 : 0.01;
        }
    }
    public static void deriveSigmoid(DMatrixRMaj vector){
        for (int i = 0; i < vector.getNumRows(); i++) {
            double p = 1 / (1 + Math.exp(-vector.data[i]));
            vector.data[i] = p*(1-p);
        }
    }
    public static void deriveCustom(DMatrixRMaj vector, CustomFunction customFunction, boolean isDerivative){
        if (customFunction == null) throw new IllegalArgumentException("Custom function cannot be null");
        if (isDerivative) {
            for (int i = 0; i < vector.getNumRows(); i++) {
                vector.data[i] = customFunction.applyAsDouble(vector.data[i]);
            }
            return;
        }
        for (int i = 0; i < vector.getNumRows(); i++) {
            double epsilon = 1e-5;
            double A = customFunction.applyAsDouble(vector.data[i] - epsilon);
            double B = customFunction.applyAsDouble(vector.data[i] + epsilon);
            vector.data[i] = (B-A)/(2*epsilon);
        }
    }
}

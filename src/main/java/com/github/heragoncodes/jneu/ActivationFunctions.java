package com.github.heragoncodes.jneu;


import org.ejml.data.DMatrixRMaj;

public class ActivationFunctions {
    public static DMatrixRMaj applyReLU(DMatrixRMaj vector){
        DMatrixRMaj result = new DMatrixRMaj(vector.getNumRows(), 1);
        for (int i = 0; i < vector.getNumRows(); i++) {
            result.data[i] = Math.max(0, vector.data[i]);
        }
        return result;
    }
    public static DMatrixRMaj applyLeakyReLU(DMatrixRMaj vector){
        DMatrixRMaj result = new DMatrixRMaj(vector.getNumRows(), 1);
        for (int i = 0; i < vector.getNumRows(); i++) {
            result.data[i] = vector.data[i] > 0 ? vector.data[i] : 0.01 * vector.data[i];
        }
        return result;
    }
    public static DMatrixRMaj applySigmoid(DMatrixRMaj vector){
        DMatrixRMaj result = new DMatrixRMaj(vector.getNumRows(), 1);
        for (int i = 0; i < vector.getNumRows(); i++) {
            result.data[i] = 1 / (1 + Math.exp(-vector.data[i]));
        }
        return result;
    }
    public static DMatrixRMaj applyCustom(DMatrixRMaj vector, CustomFunction customFunction){
        DMatrixRMaj result = new DMatrixRMaj(vector.getNumRows(), 1);
        for (int i = 0; i < vector.getNumRows(); i++) {
            result.data[i] = customFunction.applyAsDouble(vector.data[i]);
        }
        return result;
    }

    public static DMatrixRMaj deriveReLU(DMatrixRMaj vector){
        DMatrixRMaj result = new DMatrixRMaj(vector.getNumRows(), 1);
        for (int i = 0; i < vector.getNumRows(); i++) {
            result.data[i] = (vector.data[i] > 0) ? 1 : 0;
        }
        return result;
    }
    public static DMatrixRMaj deriveLeakyReLU(DMatrixRMaj vector){
        DMatrixRMaj result = new DMatrixRMaj(vector.getNumRows(), 1);
        for (int i = 0; i < vector.getNumRows(); i++) {
            result.data[i] = (vector.data[i] > 0) ? 1 : 0.01;
        }
        return result;
    }
    public static DMatrixRMaj deriveSigmoid(DMatrixRMaj vector){
        DMatrixRMaj result = new DMatrixRMaj(vector.getNumRows(), 1);
        for (int i = 0; i < vector.getNumRows(); i++) {
            double p = 1 / (1 + Math.exp(-vector.data[i]));
            result.data[i] = p*(1-p);
        }
        return result;
    }
    public static DMatrixRMaj deriveCustom(DMatrixRMaj vector, CustomFunction customFunction, boolean isDerivative){
        DMatrixRMaj result = new DMatrixRMaj(vector.getNumRows(), 1);
        if (customFunction == null) throw new IllegalArgumentException("Custom function cannot be null");
        if (isDerivative) {
            for (int i = 0; i < vector.getNumRows(); i++) {
                result.data[i] = customFunction.applyAsDouble(vector.data[i]);
            }
            return result;
        }
        for (int i = 0; i < vector.getNumRows(); i++) {
            double epsilon = 1e-5;
            double A = customFunction.applyAsDouble(vector.data[i] - epsilon);
            double B = customFunction.applyAsDouble(vector.data[i] + epsilon);
            result.data[i] = (B-A)/(2*epsilon);
        }
        return result;
    }
}

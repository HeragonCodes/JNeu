package com.github.heragoncodes.jneu;


import org.ejml.data.DMatrixRMaj;

public class LossFunctions {
    public enum Loss{
        MEAN_SQUARED,
        MEAN_ABSOLUTE,
        CROSS_ENTROPY,
        BINARY_CROSS_ENTROPY,
        CUSTOM;
    }

    public static double applyMSE(DMatrixRMaj vector, DMatrixRMaj expected){
        double sum = 0;
        for (int i = 0; i < vector.getNumRows(); i++) {
            sum += Math.pow(expected.data[i] - vector.data[i], 2);
        }
        return sum/vector.getNumRows();
    }
    public static double applyMAE(DMatrixRMaj vector, DMatrixRMaj expected){
        double sum = 0;
        for (int i = 0; i < vector.getNumRows(); i++) {
            sum += Math.abs(expected.data[i] - vector.data[i]);
        }
        return sum/vector.getNumRows();
    }
    public static double applyCE(DMatrixRMaj vector, DMatrixRMaj expected){
        double sum = 0;
        for (int i = 0; i < vector.getNumRows(); i++) {
            sum += expected.data[i] * Math.log(Math.max(vector.data[i], 1e-15));
        }
        return -sum;
    }
    public static double applyBCE(DMatrixRMaj vector, DMatrixRMaj expected){
        double sum = 0;
        for (int i = 0; i < vector.getNumRows(); i++) {
            if (expected.data[i] == 1){
                sum += Math.log(Math.max(vector.data[i], 1e-15));
            } else {
                sum += Math.log(Math.max(1 - vector.data[i], 1e-15));
            }
        }
        return -sum/vector.getNumRows();
    }
    public static double applyCustom(DMatrixRMaj vector, DMatrixRMaj expected, CustomLoss function){
        double sum = 0;
        for (int i = 0; i < vector.getNumRows(); i++) {
            sum += function.applyAsDouble(vector.data[i], expected.data[i]);
        }
        return sum/vector.getNumRows();
    }

    public static DMatrixRMaj deriveMSE(DMatrixRMaj vector, DMatrixRMaj expected){
        DMatrixRMaj result = new DMatrixRMaj(vector.getNumRows(), 1);
        for (int i = 0; i < vector.getNumRows(); i++) {
            result.data[i] = (2 * (vector.data[i] - expected.data[i]))/vector.getNumRows();
        }
        return result;
    }
    public static DMatrixRMaj deriveMAE(DMatrixRMaj vector, DMatrixRMaj expected){
        DMatrixRMaj result = new DMatrixRMaj(vector.getNumRows(), 1);
        for (int i = 0; i < vector.getNumRows(); i++) {
            result.data[i] = (double)(Double.compare(vector.data[i], expected.data[i]))/vector.getNumRows();
        }
        return result;
    }
    public static DMatrixRMaj deriveCE(DMatrixRMaj vector, DMatrixRMaj expected){
        DMatrixRMaj result = new DMatrixRMaj(vector.getNumRows(), 1);
        for (int i = 0; i < vector.getNumRows(); i++) {
            result.data[i] = -expected.data[i]/Math.max(vector.data[i], 1e-15);
        }
        return result;
    }
    public static DMatrixRMaj deriveBCE(DMatrixRMaj vector, DMatrixRMaj expected){
        DMatrixRMaj result = new DMatrixRMaj(vector.getNumRows(), 1);
        for (int i = 0; i < vector.getNumRows(); i++) {
            result.data[i] = ((vector.data[i]-expected.data[i])/Math.max(vector.data[i]*(1-vector.data[i]), 1e-15))/vector.getNumRows();
        }
        return result;
    }
    public static DMatrixRMaj deriveCustom(DMatrixRMaj vector, DMatrixRMaj expected, CustomLoss customFunction, boolean isDerivative){
        if (customFunction == null) throw new IllegalArgumentException("Custom function cannot be null");
        DMatrixRMaj result = new DMatrixRMaj(vector.getNumRows(), 1);
        if (isDerivative) {
            for (int i = 0; i < vector.getNumRows(); i++) {
                result.data[i] = customFunction.applyAsDouble(vector.data[i], expected.data[i])/vector.getNumRows();
            }
            return result;
        }
        for (int i = 0; i < vector.getNumRows(); i++) {
            double epsilon = 1e-5;
            double A = customFunction.applyAsDouble(vector.data[i] - epsilon, expected.data[i]);
            double B = customFunction.applyAsDouble(vector.data[i] + epsilon, expected.data[i]);
            result.data[i] = ((B-A)/(2*epsilon))/vector.getNumRows();
        }
        return result;
    }
}

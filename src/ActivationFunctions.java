import org.ejml.data.DMatrixRMaj;

import java.util.function.DoubleUnaryOperator;

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

    public static void applyCustom(DMatrixRMaj vector, DoubleUnaryOperator customFunction){
        for (int i = 0; i < vector.getNumRows(); i++) {
            vector.data[i] = customFunction.applyAsDouble(vector.data[i]);
        }
    }
}

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.mult.MatrixVectorMult_DDRM;

import java.io.Serializable;
import java.util.function.DoubleUnaryOperator;

public class Layer implements Serializable {

    @java.io.Serial
    private static final long serialVersionUID = 1L;

    private final DMatrixRMaj weights;
    private final DMatrixRMaj biases;

    private final DMatrixRMaj lastInput;
    private final DMatrixRMaj lastOutput;

    private Act function;
    private transient DoubleUnaryOperator customActivation = null;

    public enum Act {
        RELU,
        LEAKY_RELU,
        SIGMOID,
        CUSTOM;
    }


    public Layer(DMatrixRMaj weights, DMatrixRMaj biases, Act function){
        this.weights = weights;
        this.biases = biases;

        this.lastInput = new DMatrixRMaj(weights.getNumCols(), 1);
        this.lastOutput = new DMatrixRMaj(weights.getNumRows(), 1);

        this.function = function;
    }
    public Layer(DMatrixRMaj weights, DMatrixRMaj biases){
        this(weights, biases, Act.RELU);
    }
    public Layer(DMatrixRMaj weights, DMatrixRMaj biases, DoubleUnaryOperator customFunction){
        this(weights, biases, Act.CUSTOM);
        this.customActivation = customFunction;
    }

    public void forwardPass(){
        MatrixVectorMult_DDRM.mult(weights, lastInput, lastOutput);
        CommonOps_DDRM.addEquals(lastOutput, biases);
        switch (function) {
            case RELU -> ActivationFunctions.applyReLU(lastOutput);
            case LEAKY_RELU -> ActivationFunctions.applyLeakyReLU(lastOutput);
            case SIGMOID -> ActivationFunctions.applySigmoid(lastOutput);
            case CUSTOM -> {
                if (customActivation == null) {
                    throw new IllegalStateException("Custom activation function is null!");
                }
                ActivationFunctions.applyCustom(lastOutput, customActivation);
            }
        }
    }

    public void setActivation(Act activation){
        function = activation;
        if (activation != Act.CUSTOM) {
            this.customActivation = null;
        }
    }
    public void setCustomActivation(DoubleUnaryOperator myFunction){
        function = Act.CUSTOM;
        customActivation = myFunction;
    }
    public void setLastInput(DMatrixRMaj inputVector){
        lastInput.setTo(inputVector);
    }

    public DMatrixRMaj getLastOutput(){return lastOutput;}
}

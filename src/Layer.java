import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.mult.MatrixVectorMult_DDRM;

import java.io.Serializable;

public class Layer implements Serializable {

    @java.io.Serial
    private static final long serialVersionUID = 1L;

    private final DMatrixRMaj weights;
    private final DMatrixRMaj biases;

    private final DMatrixRMaj lastInput;
    private final DMatrixRMaj lastOutput;

    private Act function;
    private CustomFunction customActivation = null;
    private CustomFunction customDerivative = null;

    private final DMatrixRMaj weightsGradient;
    private final DMatrixRMaj biasesGradient;

    public enum Act {
        RELU,
        LEAKY_RELU,
        SIGMOID,
        CUSTOM;
    }


    public Layer(DMatrixRMaj weights, DMatrixRMaj biases, Act function){
        this.weights = weights;
        this.weightsGradient = new DMatrixRMaj(weights.getNumRows(), weights.getNumCols());
        this.biases = biases;
        this.biasesGradient = new DMatrixRMaj(biases.getNumRows(), 1);

        this.lastInput = new DMatrixRMaj(weights.getNumCols(), 1);
        this.lastOutput = new DMatrixRMaj(weights.getNumRows(), 1);

        this.function = function;
    }
    public Layer(DMatrixRMaj weights, DMatrixRMaj biases){
        this(weights, biases, Act.RELU);
    }
    public Layer(DMatrixRMaj weights, DMatrixRMaj biases, CustomFunction customFunction, CustomFunction customDerivative){
        this(weights, biases, Act.CUSTOM);
        this.customActivation = customFunction;
        this.customDerivative = customDerivative;
    }
    public Layer(DMatrixRMaj weights, DMatrixRMaj biases, CustomFunction customFunction){
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
    public DMatrixRMaj backwardPass(DMatrixRMaj dLoss){
        DMatrixRMaj delta = lastOutput.copy();
        switch (function) {
            case RELU -> ActivationFunctions.deriveReLU(delta);
            case LEAKY_RELU -> ActivationFunctions.deriveLeakyReLU(delta);
            case SIGMOID -> ActivationFunctions.deriveSigmoid(delta);
            case CUSTOM -> {
                if (customActivation == null) {
                    throw new IllegalStateException("Custom activation function is null!");
                }
                if (customDerivative == null) {
                    ActivationFunctions.deriveCustom(delta, customActivation, false);
                } else {
                    ActivationFunctions.deriveCustom(delta, customDerivative, true);
                }
            }
            default -> throw new IllegalStateException("Unknown activation function");
        }
        CommonOps_DDRM.elementMult(delta, dLoss, delta);
        CommonOps_DDRM.multAddTransB(delta, lastInput, weightsGradient);
        CommonOps_DDRM.addEquals(biasesGradient, delta);
        DMatrixRMaj errorOut = new DMatrixRMaj(lastInput.getNumRows(), 1);
        CommonOps_DDRM.multTransA(weights, delta, errorOut);
        return errorOut;
    }

    public void setActivation(Act activation){
        function = activation;
        if (activation != Act.CUSTOM) {
            this.customActivation = null;
        }
    }
    public void setCustomActivation(CustomFunction myFunction, CustomFunction myDerivative){
        function = Act.CUSTOM;
        customActivation = myFunction;
        customDerivative = myDerivative;
    }
    public void setCustomActivation(CustomFunction myFunction){
        function = Act.CUSTOM;
        customActivation = myFunction;
    }

    public void zeroGradients() {
        this.weightsGradient.zero();
        this.biasesGradient.zero();
    }
    public void updateLayer(double learningRate){
        CommonOps_DDRM.addEquals(weights, -learningRate, weightsGradient);
        CommonOps_DDRM.addEquals(biases, -learningRate, biasesGradient);
        zeroGradients();
    }

    public void setLastInput(DMatrixRMaj inputVector){
        lastInput.setTo(inputVector);
    }

    public DMatrixRMaj getLastOutput(){return lastOutput;}
}

package com.github.heragoncodes.jneu;


import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.mult.MatrixVectorMult_DDRM;

import java.io.Serializable;

public class Layer implements Serializable {

    @java.io.Serial
    private static final long serialVersionUID = 1L;

    private final DMatrixRMaj weights;
    private final DMatrixRMaj biases;

    private DMatrixRMaj lastInput;
    private final DMatrixRMaj lastOutput;
    private final DMatrixRMaj lastZ;

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
        this.lastZ = new DMatrixRMaj(weights.getNumRows(), 1);

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

    public void forwardPass(DMatrixRMaj inputVector){
        lastInput.setTo(inputVector);
        MatrixVectorMult_DDRM.mult(weights, lastInput, lastOutput);
        CommonOps_DDRM.addEquals(lastOutput, biases);
        lastZ.setTo(lastOutput);
        switch (function) {
            case RELU -> lastOutput.setTo(ActivationFunctions.applyReLU(lastOutput));
            case LEAKY_RELU -> lastOutput.setTo(ActivationFunctions.applyLeakyReLU(lastOutput));
            case SIGMOID -> lastOutput.setTo(ActivationFunctions.applySigmoid(lastOutput));
            case CUSTOM -> {
                if (customActivation == null) {
                    throw new IllegalStateException("Custom activation function is null!");
                }
                lastOutput.setTo(ActivationFunctions.applyCustom(lastOutput, customActivation));
            }
        }
    }
    public DMatrixRMaj backwardPass(DMatrixRMaj dLoss){
        DMatrixRMaj delta;
        switch (function) {
            case RELU -> delta = ActivationFunctions.deriveReLU(lastZ);
            case LEAKY_RELU -> delta = ActivationFunctions.deriveLeakyReLU(lastZ);
            case SIGMOID -> delta = ActivationFunctions.deriveSigmoid(lastZ);
            case CUSTOM -> {
                if (customActivation == null) {
                    throw new IllegalStateException("Custom activation function is null!");
                }
                if (customDerivative == null) {
                    delta = ActivationFunctions.deriveCustom(lastZ, customActivation, false);
                } else {
                    delta = ActivationFunctions.deriveCustom(lastZ, customDerivative, true);
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

    public DMatrixRMaj getLastOutput(){return lastOutput;}
}

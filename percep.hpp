#pragma once
#include <vector>

class Perceptron {
private:
    std::vector<float> weights;
    float bias;
    float learningRate;

public:
    Perceptron(int inputSize, float lr = 0.1f);

    int activate(float sum);
    int predict(const std::vector<float>& inputs);
    void train(const std::vector<float>& inputs, int target);
    void printWeights();
};

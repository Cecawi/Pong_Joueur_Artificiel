// LinearModel.h
#pragma once
#include <vector>

class LinearModel
{
    private:
        std::vector<float> weights;
        float bias;
        float learningRate;

    public:
        LinearModel(int inputSize, float lr = 0.01f);
        float predict(const std::vector<float>& x);
        void train(const std::vector<std::vector<float>>& X,
                   const std::vector<float>& y,
                   int epochs);
};
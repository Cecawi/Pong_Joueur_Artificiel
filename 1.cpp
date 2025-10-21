// LinearModel.cpp
#include "1.hpp"
#include <cmath>
#include <iostream>
#include <vector>

LinearModel::LinearModel(int inputSize, float lr)
    : weights(inputSize, 0.0f), bias(0.0f), learningRate(lr) {}

float LinearModel::predict(const std::vector<float>& x) {
    float sum = bias;
    for (size_t i = 0; i < x.size(); ++i)
        sum += weights[i] * x[i];
    return sum;
}

//pas mal

void LinearModel::train(const std::vector<std::vector<float>>& X,
    const std::vector<float>& y, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < X.size(); ++i) {
            float pred = predict(X[i]);
            float error = y[i] - pred;

            // mise à jour des poids
            for (size_t j = 0; j < weights.size(); ++j)
                weights[j] += learningRate * error * X[i][j];

            bias += learningRate * error;
        }
    }
}

int main() {


std::vector<std::vector<float>> X = { {1}, {2}, {3}, {4} };
std::vector<float> y = { 3, 5, 7, 9 };  // 2x + 1

LinearModel model(1, 0.01f);
model.train(X, y, 1000);

// test
for (auto& x : X)
std::cout << "x=" << x[0] << " -> pred=" << model.predict(x) << std::endl;


    return 0;
}

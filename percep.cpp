// #include <iostream>
// #include <vector>

// class Perceptron {
// private:
//     std::vector<float> weights; // weights for each input
//     float bias;                 // bias
//     float learningRate;

// public:
//     Perceptron(int inputSize, float lr = 0.1) {
//         learningRate = lr;
//         weights.resize(inputSize, 0.0f); // initialize weights to 0
//         bias = 0.0f;                     // initialize bias to 0
//     }

//     // Simple step activation function
//     int activate(float sum) {
//         return (sum >= 0) ? 1 : 0;
//     }

//     // Make a prediction
//     int predict(const std::vector<float>& inputs) {
//         float sum = bias; // start with bias
//         for (size_t i = 0; i < inputs.size(); ++i)
//             sum += weights[i] * inputs[i]; // weighted sum
//         return activate(sum);
//     }

//     // Train on a single example
//     void train(const std::vector<float>& inputs, int target) {
//         int output = predict(inputs);
//         int error = target - output;

//         // Update weights and bias
//         for (size_t i = 0; i < weights.size(); ++i)
//             weights[i] += learningRate * error * inputs[i];
//         bias += learningRate * error;
//     }

//     void print() {
//         std::cout << "Weights: ";
//         for (float w : weights)
//             std::cout << w << " ";
//         std::cout << "| Bias: " << bias << "\n";
//     }
// };

// int main() {
//     // Example: OR gate
//     std::vector<std::vector<float>> X = {{0,0},{0,1},{1,0},{1,1}};
//     std::vector<int> Y = {0,1,1,1};

//     Perceptron p(2, 0.1); // 2 inputs, learning rate 0.1

//     // Train for 10 epochs
//     for (int epoch = 0; epoch < 10; ++epoch) {
//         for (size_t i = 0; i < X.size(); ++i)
//             p.train(X[i], Y[i]);
//     }

//     p.print();

//     // Test predictions
//     for (size_t i = 0; i < X.size(); ++i)
//         std::cout << "Input: (" << X[i][0] << "," << X[i][1] 
//                   << ") => Predictionnnnn: " << p.predict(X[i]) << "\n";

//     return 0;
// }


// ancien code

#include <iostream>
#include <vector>

class Perceptron {
private:
    std::vector<float> weights;
    float bias;
    float learningRate;

public:
    Perceptron(int inputSize, float lr = 0.1f) {
        weights.resize(inputSize, 0.0f); //  weights 0 as usual
        bias = 0.0f;
        learningRate = lr;
    }

    int activate(float sum) {
        return (sum >= 0) ? 1 : 0;
    }

    int predict(const std::vector<float>& inputs) {
        float sum = bias;
        for (size_t i = 0; i < inputs.size(); ++i)
            sum += weights[i] * inputs[i];
        return activate(sum);
    }

    void train(const std::vector<float>& inputs, int target) {
        int output = predict(inputs);
        int error = target - output;

        for (size_t i = 0; i < weights.size(); ++i)
            weights[i] += learningRate * error * inputs[i];
        bias += learningRate * error;
    }

    void printWeights() {
        std::cout << "Weights: ";
        for (float w : weights) std::cout << w << " ";
        std::cout << "| Bias:::: " << bias << "\n";
    }
};

int main() {
    // Dataset linéairement séparable ducoup
    std::vector<std::vector<float>> X = {
        {0,0},{0,1},{1,0},{1,1},{0.5,0.5},{1,0.5}
    };
    std::vector<int> Y = {0,0,0,1,0,1}; // AND-like pattern

    // Le split training 4 points, testing 2 points
    std::vector<std::vector<float>> X_train = {X[0], X[1], X[2], X[3]};
    std::vector<int> Y_train = {Y[0], Y[1], Y[2], Y[3]};
    std::vector<std::vector<float>> X_test = {X[4], X[5]};
    std::vector<int> Y_test = {Y[4], Y[5]};

    Perceptron p(2, 0.1);

    // le training
    for (int epoch = 0; epoch < 10; ++epoch) {
        for (size_t i = 0; i < X_train.size(); ++i)
            p.train(X_train[i], Y_train[i]);
        std::cout << "Epoch " << epoch+1 << ": ";
        p.printWeights();
    }

    // Testing on unseen data
    std::cout << "\nTesting on unseen data:\n";
    for (size_t i = 0; i < X_test.size(); ++i) {
        int pred = p.predict(X_test[i]);
        std::cout << "Input: (" << X_test[i][0] << "," << X_test[i][1] << ") => Prediction: "
                  << pred << " | Target: " << Y_test[i] << "\n";
    }

    return 0;
}

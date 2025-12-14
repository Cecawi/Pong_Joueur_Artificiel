#pragma once 
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <functional>

//================ RBF Network Class ===================
class RBF {

    public:
        //constructeur : n_hidden = nombre de RBF neurons, input_size, output_size
        RBF(int input_size, int n_hidden, int output_size);

        //propagation avant
        std::vector<double> predict(const std::vector<double>& input);

        //entraînement simple pseudo-inverse (RBF typique)
        void train(const std::vector<std::vector<double>>& X,
                const std::vector<std::vector<double>>& Y);
    private:
        //calcule la distance euclidienne entre input et centre k
        double rbf_output(const std::vector<double>& input, int k);
    void kmeans_clustering(const std::vector<std::vector<double>>& X);
    void compute_sigmas();
    void initialize_weights();
        };
                




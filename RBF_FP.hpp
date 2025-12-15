#pragma once

#include <vector>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <cassert>

// AJOUT : Inclusion d'Eigen ici pour les types dans les fonctions friend/utilitaires si nécessaire
// Mais il est plus propre de l'inclure uniquement si nécessaire, ici, nous le laissons
// dans le .cpp pour une compilation plus rapide, mais déclarons les fonctions
// friend et les membres privés ici.

// Pour éviter de dépendre d'Eigen dans l'interface (.hpp), nous ne laissons que les déclarations.

class RBF
{
private:
    int input_size;          
    int hidden_size;         
    int output_size;         
    
    std::vector<std::vector<double>> W; 
    std::vector<std::vector<double>> centers; 
    std::vector<double> sigmas; 
    std::vector<double> hidden_activations; 
    
    double gaussian_activation(const std::vector<double>& input, int center_index) const;
    void initialize_centers_and_sigmas(const std::vector<std::vector<double>>& all_samples_inputs);
    void propagate(const std::vector<double>& input);

public:
    RBF(const std::vector<int>& sizes);

    void train
    (
        const std::vector<std::vector<double>>& all_samples_inputs,
        const std::vector<std::vector<double>>& all_samples_expected_outputs,
        int num_iter, 
        double alpha, 
        bool use_sgd_for_weights 
    );

    std::vector<double> predict(const std::vector<double>& inputs);
    
    int getInputSize() const { return input_size; }
    int getOutputSize() const { return output_size; }
    int getHiddenSize() const { return hidden_size; }
};

// Fonctions d'export C-style (extern "C")
extern "C"
{
    // Déclarations inchangées
    #if WIN32
    #define DLLEXPORT __declspec(dllexport)
    #else
    #define DLLEXPORT
    #endif

    DLLEXPORT void* create_rbf(const int* sizes, int layers_count);
    DLLEXPORT void destroy_rbf(void* handle);
    
    DLLEXPORT int train_rbf
    (
        void* handle, const double* X_flat, const double* Y_flat, 
        int samples, int input_size, int output_size, 
        int is_classification, int num_iter, double alpha
    );
    
    DLLEXPORT int predict_rbf
    (
        void* handle, const double* input, int input_size,
        double* out_buffer, int output_size, int is_classification
    );
}
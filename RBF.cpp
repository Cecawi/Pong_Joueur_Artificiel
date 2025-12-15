#if WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include "RBF.hpp"

// Définition de la fonction Gaussienne
// phi_j(x) = exp( - ||x - mu_j||^2 / (2 * sigma_j^2) )
double RBF::gaussian_activation(const std::vector<double>& input, int center_index) const
{
    double squared_distance = 0.0;
    const std::vector<double>& center = centers[center_index];
    
    for (size_t i = 0; i < input.size(); ++i)
    {
        squared_distance += std::pow(input[i] - center[i], 2);
    }
    
    double sigma_sq = sigmas[center_index] * sigmas[center_index];
    // Éviter la division par zéro
    if (sigma_sq < 1e-9) return 0.0; 

    return std::exp(-squared_distance / (2.0 * sigma_sq));
}

// K-means simplifié pour l'initialisation des centres et sigmas
void RBF::initialize_centers_and_sigmas(const std::vector<std::vector<double>>& all_samples_inputs)
{
    int N = all_samples_inputs.size();
    if (N == 0 || hidden_size == 0) return;

    centers.resize(hidden_size, std::vector<double>(input_size));
    srand(time(0)); // Utilisation de rand() comme dans PMC.cpp
    for (int j = 0; j < hidden_size; ++j)
    {
        int k = rand() % N;
        centers[j] = all_samples_inputs[k];
    }


    // d_max est la distance maximale entre les centres initiaux.
    double d_max = 0.0;
    for (int j1 = 0; j1 < hidden_size; ++j1)
    {
        for (int j2 = j1 + 1; j2 < hidden_size; ++j2)
        {
            double dist_sq = 0.0;
            for (int i = 0; i < input_size; ++i)
            {
                dist_sq += std::pow(centers[j1][i] - centers[j2][i], 2);
            }
            d_max = std::max(d_max, std::sqrt(dist_sq));
        }
    }
    
    // Détermination de l'écart-type global (l'approche simple)
    double sigma_global = (d_max == 0.0) ? 1.0 : d_max / std::sqrt(2.0 * hidden_size);

    // Assigner le même sigma à tous les neurones cachés (approche courante)
    sigmas.assign(hidden_size, sigma_global);
}

// Fonction de propagation : calcule les activations de la couche cachée
void RBF::propagate(const std::vector<double>& input)
{
    // On ignore le biais sur l'entrée, car la RBF est non linéaire
    assert(input.size() == (size_t)input_size);
    
    hidden_activations.resize(hidden_size);
    for (int j = 0; j < hidden_size; ++j)
    {
        hidden_activations[j] = gaussian_activation(input, j);
    }
}

// Constructeur
RBF::RBF(const std::vector<int>& sizes)
{
    if (sizes.size() < 2)
    {
        throw std::runtime_error("RBF() : nécessite au moins la taille d'entrée et de sortie.");
    }
    input_size = sizes[0];
    hidden_size = (sizes.size() == 2) ? 0 : sizes[1]; // S'il n'y a pas de taille cachée spécifiée
    output_size = sizes.back();
    
    // Initialisation des poids W (avec une petite valeur aléatoire)
    // W[j][k] : poids du j-ième centre vers la k-ième sortie
    W.resize(hidden_size + 1, std::vector<double>(output_size)); // +1 pour le biais du RBF
    
    srand(time(0));
    for (int j = 0; j <= hidden_size; ++j)
    {
        for (int k = 0; k < output_size; ++k)
        {
            // Initialisation des poids
            W[j][k] = ((double)rand() / RAND_MAX) * 0.1 - 0.05; // [-0.05, 0.05]
        }
    }
    
}

// Entraînement : K-means puis (Pseudo-Inverse ou SGD)
void RBF::train
(
    const std::vector<std::vector<double>>& all_samples_inputs,
    const std::vector<std::vector<double>>& all_samples_expected_outputs,
    int num_iter,
    double alpha,
    bool use_sgd_for_weights
)
{
    if (all_samples_inputs.empty()) return;
    int N = all_samples_inputs.size();

    // Étape 1 : Détermination des centres et sigmas (RBF est un modèle hybride)
    initialize_centers_and_sigmas(all_samples_inputs); 

    // Étape 2 : Entraînement des poids de sortie W (par SGD comme dans le PMC)
    for (int iter = 0; iter < num_iter; ++iter)
    {
        // Choix aléatoire d'un échantillon (Stochastic Gradient Descent)
        int k = rand() % N;
        const std::vector<double>& inputs_k = all_samples_inputs[k];
        const std::vector<double>& expected_outputs_k = all_samples_expected_outputs[k];

        // Propagation avant : calcule les hidden_activations
        propagate(inputs_k); 

        // Calcul des sorties
        std::vector<double> actual_outputs(output_size, 0.0);
        // Ajout du biais (W[0][k])
        for (int k = 0; k < output_size; ++k)
        {
            actual_outputs[k] += W[0][k]; // Biais 
        }

        // Somme pondérée des activations RBF 
        for (int j = 0; j < hidden_size; ++j)
        {
            for (int k = 0; k < output_size; ++k)
            {
                actual_outputs[k] += W[j+1][k] * hidden_activations[j];
            }
        }
        
        // Calcul des erreurs (deltas) pour la couche de sortie
        std::vector<double> deltas(output_size);
        for (int k = 0; k < output_size; ++k)
        {
            deltas[k] = actual_outputs[k] - expected_outputs_k[k];
        }

        // Mise à jour des poids (méthode de la descente de gradient, règle Delta)
        for (int k = 0; k < output_size; ++k)
        {
            // Mise à jour du biais (W[0][k])
            W[0][k] -= alpha * deltas[k];
            
            // Mise à jour des autres poids (W[j+1][k])
            for (int j = 0; j < hidden_size; ++j)
            {
                W[j+1][k] -= alpha * hidden_activations[j] * deltas[k];
            }
        }
    }
}

// Prédiction
std::vector<double> RBF::predict(const std::vector<double>& inputs)
{
    propagate(inputs);

    std::vector<double> out(output_size, 0.0);
    
    // Ajout du biais (W[0][k])
    for (int k = 0; k < output_size; ++k)
    {
        out[k] += W[0][k];
    }
    
    // Somme pondérée des activations RBF 
    for (int j = 0; j < hidden_size; ++j)
    {
        for (int k = 0; k < output_size; ++k)
        {
            out[k] += W[j+1][k] * hidden_activations[j];
        }
    }

    return out;
}

extern "C"
{
    
    DLLEXPORT void* create_rbf(const int* sizes, int layers_count)
    {
        // layers_count doit être 2 (entrée, sortie) ou 3 (entrée, caché, sortie)
        if (sizes == nullptr || layers_count < 2 || layers_count > 3) return nullptr;
        try {
            std::vector<int> npl(sizes, sizes + layers_count);
            RBF* net = new RBF(npl);
            return static_cast<void*>(net);
        } catch(...) { return nullptr; }
    }
    
    DLLEXPORT void destroy_rbf(void* handle)
    {
        if (handle) delete static_cast<RBF*>(handle);
    }
    
    // train_rbf (simplifié, prend les mêmes arguments que train_pmc)
    DLLEXPORT int train_rbf
    (
        void* handle, const double* X_flat, const double* Y_flat, 
        int samples, int input_size, int output_size, 
        int is_classification, int num_iter, double alpha
    )
    {
        if (!handle || samples <= 0 || input_size <= 0 || output_size <= 0 || !X_flat || !Y_flat) return -1;
        
        RBF* net = static_cast<RBF*>(handle);
        // Vérification de la cohérence des tailles
        if (input_size != net->getInputSize() || output_size != net->getOutputSize()) return -2; 
        
        std::vector<std::vector<double>> all_inputs(samples, std::vector<double>(input_size));
        std::vector<std::vector<double>> all_outputs(samples, std::vector<double>(output_size));
        
        for(int i = 0 ; i < samples ; ++i)
        {
            for(int j = 0 ; j < input_size ; ++j)
                all_inputs[i][j] = X_flat[i * input_size + j];
            for(int j = 0 ; j < output_size ; ++j)
                all_outputs[i][j] = Y_flat[i * output_size + j];
        }
        
        try {
            // is_classification est ignoré car RBF utilise une sortie linéaire pour la régression/classification
            net->train(all_inputs, all_outputs, num_iter, alpha, true); // Utilise SGD par défaut
        } catch(...) { return -3; }
        
        return 0;
    }
    
    // predict_rbf
    DLLEXPORT int predict_rbf
    (
        void* handle, const double* input, int input_size,
        double* out_buffer, int output_size, int is_classification
    )
    {
        if (!handle || !input || !out_buffer || input_size <= 0 || output_size <= 0) return -1;
        
        RBF* net = static_cast<RBF*>(handle);
        if (input_size != net->getInputSize() || output_size != net->getOutputSize()) return -2;

        std::vector<double> vin(input, input + input_size);
        try {
            std::vector<double> vout = net->predict(vin);
            if((int)vout.size() != output_size) return -3;
            for(int i = 0 ; i < output_size ; ++i)
                out_buffer[i] = vout[i];
        } catch(...) { return -4; }
        
        return 0;
    }
}
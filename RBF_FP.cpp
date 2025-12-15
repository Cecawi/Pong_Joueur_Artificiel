#if WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

// Inclusion d'Eigen pour les opérations matricielles
#include <Eigen/Dense>
#include "RBF.hpp"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <limits>


// --- Fonctions Utilitaires ---

// Fonction pour convertir std::vector<std::vector<double>> en matrice Eigen
Eigen::MatrixXd vectorToEigen(const std::vector<std::vector<double>>& vec)
{
    if (vec.empty() || vec[0].empty()) return Eigen::MatrixXd(0, 0);

    Eigen::MatrixXd mat(vec.size(), vec[0].size());
    for (size_t i = 0; i < vec.size(); ++i)
    {
        for (size_t j = 0; j < vec[i].size(); ++j)
        {
            mat(i, j) = vec[i][j];
        }
    }
    return mat;
}

// Fonction pour convertir matrice Eigen en std::vector<std::vector<double>>
std::vector<std::vector<double>> eigenToVector(const Eigen::MatrixXd& mat)
{
    std::vector<std::vector<double>> vec(mat.rows(), std::vector<double>(mat.cols()));
    for (int i = 0; i < mat.rows(); ++i)
    {
        for (int j = 0; j < mat.cols(); ++j)
        {
            vec[i][j] = mat(i, j);
        }
    }
    return vec;
}

// Fonction utilitaire : Calcule la distance euclidienne carrée
double squared_euclidean_distance(const std::vector<double>& v1, const std::vector<double>& v2)
{
    double dist_sq = 0.0;
    for (size_t i = 0; i < v1.size(); ++i)
    {
        dist_sq += std::pow(v1[i] - v2[i], 2);
    }
    return dist_sq;
}


// --- Pseudo-Inverse M-P (par Décomposition SVD) ---
// Note : Eigen fournit une méthode élégante pour la Pseudo-Inverse
Eigen::MatrixXd pseudoInverse(const Eigen::MatrixXd& a, double epsilon = std::numeric_limits<double>::epsilon())
{
    // 1. Décomposition en valeurs singulières
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
    
    // 2. Calcul de la tolérance
    double tolerance = epsilon * std::max(a.cols(), a.rows()) * svd.singularValues().array().abs()(0);
    
    // 3. Calcul du vecteur des valeurs singulières inversées ou nulles
    //    On force l'évaluation en VectorXd avec .matrix()
    Eigen::VectorXd singularValuesInv = (svd.singularValues().array().abs() > tolerance)
        .select(svd.singularValues().array().abs().cwiseInverse(), 0)
        .matrix(); // <-- Correction ici: force l'évaluation en Vector/Matrix

    // 4. Reconstruction de la matrice (V * Sigma^+ * U^T)
    return svd.matrixV() * singularValuesInv.asDiagonal() * svd.matrixU().transpose();
}
// --- Méthodes RBF ---

double RBF::gaussian_activation(const std::vector<double>& input, int center_index) const
{
    double squared_distance = 0.0;
    const std::vector<double>& center = centers[center_index];
    
    for (size_t i = 0; i < input.size(); ++i)
    {
        squared_distance += std::pow(input[i] - center[i], 2);
    }
    
    double sigma_sq = sigmas[center_index] * sigmas[center_index];
    if (sigma_sq < 1e-9) return 0.0; 

    return std::exp(-squared_distance / (2.0 * sigma_sq));
}

// --- Algorithme de Lloyd (K-Means) ---
void RBF::initialize_centers_and_sigmas(const std::vector<std::vector<double>>& all_samples_inputs)
{
    int N = all_samples_inputs.size();
    if (N == 0 || hidden_size == 0) return;

    // 1. Initialisation des centres
    centers.resize(hidden_size, std::vector<double>(input_size));
    srand(time(0));
    for (int j = 0; j < hidden_size; ++j)
    {
        int k = rand() % N;
        centers[j] = all_samples_inputs[k];
    }
    
    // Répéter l'Algorithme de Lloyd (K-Means)
    std::vector<int> assignments(N);
    const int MAX_ITER = 50;
    
    for (int iter = 0; iter < MAX_ITER; ++iter)
    {
        bool changed = false;

        // Étape 2 : Affectation des échantillons aux centres les plus proches
        for (int n = 0; n < N; ++n)
        {
            double min_dist = std::numeric_limits<double>::max();
            int best_center = -1;

            for (int k = 0; k < hidden_size; ++k)
            {
                double dist = squared_euclidean_distance(all_samples_inputs[n], centers[k]);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    best_center = k;
                }
            }
            if (assignments[n] != best_center)
            {
                assignments[n] = best_center;
                changed = true;
            }
        }
        
        if (!changed && iter > 0) break;

        // Étape 1 : Mise à jour des centres (calcul de la moyenne)
        std::vector<std::vector<double>> new_centers(hidden_size, std::vector<double>(input_size, 0.0));
        std::vector<int> counts(hidden_size, 0);

        for (int n = 0; n < N; ++n)
        {
            int k = assignments[n];
            if (k != -1)
            {
                for (int i = 0; i < input_size; ++i)
                {
                    new_centers[k][i] += all_samples_inputs[n][i];
                }
                counts[k]++;
            }
        }

        for (int k = 0; k < hidden_size; ++k)
        {
            if (counts[k] > 0)
            {
                for (int i = 0; i < input_size; ++i)
                {
                    centers[k][i] = new_centers[k][i] / counts[k];
                }
            }
        }
    }
    
    // 3. Calcul de Sigma (méthode de la distance maximale)
    double d_max = 0.0;
    for (int j1 = 0; j1 < hidden_size; ++j1)
    {
        for (int j2 = j1 + 1; j2 < hidden_size; ++j2)
        {
            d_max = std::max(d_max, std::sqrt(squared_euclidean_distance(centers[j1], centers[j2])));
        }
    }
    
    double sigma_global = (d_max == 0.0) ? 1.0 : d_max / std::sqrt(2.0 * hidden_size);
    sigmas.assign(hidden_size, sigma_global);
}

void RBF::propagate(const std::vector<double>& input)
{
    assert(input.size() == (size_t)input_size);
    
    hidden_activations.resize(hidden_size);
    for (int j = 0; j < hidden_size; ++j)
    {
        hidden_activations[j] = gaussian_activation(input, j);
    }
}

RBF::RBF(const std::vector<int>& sizes)
{
    if (sizes.size() < 2)
    {
        throw std::runtime_error("RBF() : nécessite au moins la taille d'entrée et de sortie.");
    }
    input_size = sizes[0];
    hidden_size = (sizes.size() == 2) ? 0 : sizes[1];
    output_size = sizes.back();
    
    // Les poids W seront mis à jour par la Pseudo-Inverse,
    // mais la structure est initialisée
    W.resize(hidden_size + 1, std::vector<double>(output_size)); 
}

// --- MISE À JOUR : Entraînement par Lloyd + Pseudo-Inverse ---
void RBF::train
(
    const std::vector<std::vector<double>>& all_samples_inputs,
    const std::vector<std::vector<double>>& all_samples_expected_outputs,
    int num_iter, // PARAMÈTRE IGNORÉ
    double alpha, // PARAMÈTRE IGNORÉ
    bool use_sgd_for_weights // PARAMÈTRE IGNORÉ
)
{
    if (all_samples_inputs.empty()) return;
    int N = all_samples_inputs.size();

    // Étape 1 : Détermination des centres et sigmas (par Lloyd)
    initialize_centers_and_sigmas(all_samples_inputs); 

    // Étape 2 : Calcul des poids de sortie W par Pseudo-Inverse (Moindres Carrés Analytique)
    
    // 2.1 Construction de la matrice Phi (N x K+1)
    int K_plus_bias = hidden_size + 1;
    Eigen::MatrixXd Phi(N, K_plus_bias);

    for (int n = 0; n < N; ++n)
    {
        // Colonne 0 : Biais
        Phi(n, 0) = 1.0; 
        
        // Colonnes 1 à K : Activations RBF
        for (int j = 0; j < hidden_size; ++j)
        {
            double squared_dist = squared_euclidean_distance(all_samples_inputs[n], centers[j]);
            double sigma_sq = sigmas[j] * sigmas[j];
            
            // Assurez-vous que sigma_sq n'est pas zéro pour éviter NaN
            Phi(n, j + 1) = (sigma_sq < 1e-9) ? 0.0 : std::exp(-squared_dist / (2.0 * sigma_sq));
        }
    }
    
    // 2.2 Construction de la matrice Y (N x OutputSize)
    Eigen::MatrixXd Y = vectorToEigen(all_samples_expected_outputs);

    // 2.3 Calcul de W par Pseudo-Inverse (Moore-Penrose)
    // W = Phi^+ * Y
    
    // NOTE: Eigen::MatrixXd::colPivHouseholderQr().solve() est souvent utilisé pour les moindres carrés,
    // mais la Pseudo-Inverse via SVD est plus robuste contre les matrices mal conditionnées (le cas RBF).
    
    Eigen::MatrixXd W_mat = pseudoInverse(Phi) * Y;

    // 2.4 Mise à jour des poids internes
    W = eigenToVector(W_mat);
}


// --- Fonctions d'export C-style (inchangées dans leur structure) ---
// Les fonctions train_rbf ci-dessous ne doivent plus se soucier de num_iter et alpha,
// car RBF::train les ignore maintenant.

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
    
    DLLEXPORT int train_rbf
    (
        void* handle, const double* X_flat, const double* Y_flat, 
        int samples, int input_size, int output_size, 
        int is_classification, int num_iter, double alpha
    )
    {
        if (!handle || samples <= 0 || input_size <= 0 || output_size <= 0 || !X_flat || !Y_flat) return -1;
        
        RBF* net = static_cast<RBF*>(handle);
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
            // Les paramètres num_iter et alpha sont ignorés car nous utilisons la Pseudo-Inverse
            net->train(all_inputs, all_outputs, 0, 0.0, false); 
        } catch(...) { return -3; }
        
        return 0;
    }
    
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
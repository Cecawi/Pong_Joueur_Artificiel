#pragma once

#include <vector>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <ctime>

class RBF
{
private:
    int input_size;          // Nombre de neurones d'entrée
    int hidden_size;         // Nombre de neurones cachés (centres RBF)
    int output_size;         // Nombre de neurones de sortie
    
    // Poids de la couche de sortie : W[j][k] est le poids entre le centre j et la sortie k
    // W[j][0] est le biais si l'implémentation utilise un biais sur la sortie.
    // Pour une sortie linéaire, on utilise généralement W[j][k] où j va de 0 à hidden_size.
    // Ici, nous allons simplifier : W[j][k] où j = centre, k = sortie.
    std::vector<std::vector<double>> W; 

    // Centres des fonctions de base radiales (déterminés par K-means)
    // centers[j][i] : valeur de la i-ième dimension pour le j-ième centre
    std::vector<std::vector<double>> centers; 
    
    // Écarts-types (largeurs) des fonctions de base radiales : sigma[j]
    std::vector<double> sigmas; 

    // Stockage temporaire des activations RBF
    std::vector<double> hidden_activations; 
    
    // Fonction d'activation de la couche cachée (Gaussienne)
    double gaussian_activation(const std::vector<double>& input, int center_index) const;

    // K-means pour l'initialisation des centres
    void initialize_centers_and_sigmas(const std::vector<std::vector<double>>& all_samples_inputs);

    // Fonction de propagation
    void propagate(const std::vector<double>& input);

public:
    // Constructeur
    // sizes[0]: input_size, sizes[1]: hidden_size (centres), sizes[2]: output_size
    RBF(const std::vector<int>& sizes);

    // Entraînement : Calcul des centres (K-means) et des poids de sortie (pseudo-inverse ou SGD)
    void train
    (
        const std::vector<std::vector<double>>& all_samples_inputs,
        const std::vector<std::vector<double>>& all_samples_expected_outputs,
        int num_iter,
        double alpha, // Pas d'apprentissage pour SGD (si utilisé)
        bool use_sgd_for_weights = true // Vrai pour SGD, Faux pour Pseudo-Inverse
    );

    // Prédiction
    std::vector<double> predict(const std::vector<double>& inputs);
    
    // Getters pour chargement/sauvegarde
    int getInputSize() const { return input_size; }
    int getOutputSize() const { return output_size; }
    int getHiddenSize() const { return hidden_size; }
    std::vector<std::vector<double>> getWeights() const { return W; }
    std::vector<std::vector<double>> getCenters() const { return centers; }
    std::vector<double> getSigmas() const { return sigmas; }
    
    // Setters pour chargement
    void setWeights(const std::vector<std::vector<double>>& newW) { W = newW; }
    void setCenters(const std::vector<std::vector<double>>& newC) { centers = newC; }
    void setSigmas(const std::vector<double>& newS) { sigmas = newS; }
};

// Fonctions d'export C-style (similaires à PMC.cpp)
extern "C"
{
    // [Implémentations des fonctions d'export comme create_rbf, destroy_rbf, train_rbf, predict_rbf]
}
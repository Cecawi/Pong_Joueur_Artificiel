#pragma once

/////DEMANDER POUR LES INCLUDES

#include <vector>
#include <cmath>//tanh
#include <cstdlib>//rand(), RAND_MAX
#include <cassert>//assert
#include <stdexcept>//runtime_error

/////REGARDER LA OU Y A !!!!!/////

class PMC
{
    private : 
        std::vector<int> d;//liste taille des couches, d[l] : nombre de neurones dans la couche l
        int L;//L : nombre de couches cachées (la dernière est L)
        std::vector<std::vector<std::vector<double>>> W;//poids, W[l][i][j] : poids de la couche l liant le neurone i de la couche l - 1 au neurone j de la couche l
        std::vector<std::vector<double>> X;/////activations ; OK : //X[l][j] : valeur du neurone j de la couche l
        std::vector<std::vector<double>> deltas;/////erreurs[l][j]
    
    public : 
        //constructeur
        PMC(const std::vector<int>& npl);//npl : neurons_per_layer (y compris la couche d’entrée)

        //calcule les activations de toutes les couches
        void propagate(const std::vector<double>& inputs, bool is_classification);

        //entraînement : rétropropagation du gradient stochastique
        void train
        (
        	const std::vector<std::vector<double>>& all_samples_inputs,
        	const std::vector<std::vector<double>>& all_samples_expected_outputs,
        	bool is_classification,
        	int num_iter,
        	double alpha//pas d'apprentissage
        );

        //prédiction
        std::vector<double> predict(const std::vector<double>& inputs, bool is_classification);

        //getters
        int getInputSize() const
        {
            return d.empty() ? 0 : d[0];
        }

        int getOutputSize() const
        {
            return d.empty() ? 0 : d[L];
        }
};
#include "PMC.hpp"

/////REGARDER LA OU Y A !!!!!/////

//constructeur
PMC::PMC(const std::vector<int>& npl)//npl : neurons_per_layer (y compris la couche d’entrée)
{
    d = npl;//copie
    L = npl.size() - 1;//dernière couche

    //W[l][i][j] : poids de la couche l liant le neurone i de la couche l - 1 au neurone j de la couche l
    //initialisation des poids
    W.resize(d.size());
    for(size_t l = 0 ; l < d.size() ; ++l)
    {
        if (l == 0)
        {
            W[l].resize(1);
            continue;//pas de poids pour la couche d'entrée
        }

        W[l].resize(d[l - 1] + 1);//+1 pour biais, ignore l==0

        for(size_t i = 0 ; i <= d[l - 1] ; ++i)
        {
            W[l][i].resize(d[l] + 1);//+1 pour biais
            for(size_t j = 0 ; j <= d[l] ; ++j)
            {
                if(j == 0)
                {
                    W[l][i][j] = 0.0;//biais
                }
                else
                {
                    /////std::mt19937
                    W[l][i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;//[-1,1]
                }
            }
        }
    }

    //initialisation des activations et deltas
    X.resize(d.size());
    deltas.resize(d.size());
    for(size_t l = 0 ; l < d.size() ; ++l)
    {
        X[l].resize(d[l] + 1);//+1 pour biais
        deltas[l].resize(d[l] + 1);
        for(size_t j = 0 ; j <= d[l] ; ++j)
        {
            X[l][j] = (j == 0) ? 1.0 : 0.0;//biais = 1.0
            deltas[l][j] = 0.0;/////erreurs pour la rétropropagation
        }
    }
}

//calcule les activations de toutes les couches
void PMC::propagate(const std::vector<double>& inputs, bool is_classification)
{
    //vérification si même nombre d'entrées que nombre de neurones d'entrée
    assert(inputs.size() == (size_t)d[0]);

    //mise à jour des activations de la couche d'entrées 0
    for(size_t j = 0 ; j < inputs.size() ; ++j)
    {
        //X[0][0] = 1.0 (le biais)
        X[0][j + 1] = inputs[j];//+1 pour ignorer le biais
    }

    //propagation couche par couche
    for(size_t l = 1 ; l <= (size_t)L ; ++l)
    {

        //pour chaque neurone j de la couche l (en ignorant j=0 = biais)
        for(size_t j = 1 ; j <= (size_t)d[l] ; ++j)
        {
            double signal = 0.0;

            //produit somme des poids W[l][i][j] * X[l-1][i]
            //i parcourt tous les neurones (y compris le biais i=0)
            for(size_t i = 0 ; i <= (size_t)d[l - 1] ; ++i)
            {
                signal += W[l][i][j] * X[l - 1][i];
            }

            double x = signal;
            //activation : tanh si classification ou couche cachée
            if(is_classification || l != L)
            {
                x = std::tanh(signal);
            }

            //on stocke l'activation du neurone j
            X[l][j] = x;
        }
    }
}

//entraînement : rétropropagation du gradient stochastique
void PMC::train
(
const std::vector<std::vector<double>>& all_samples_inputs,
const std::vector<std::vector<double>>& all_samples_expected_outputs,
bool is_classification,
int num_iter,
double alpha//pas d'apprentissage
)
{
    //vérifir si même nombre d’entrées que de sorties
    if(all_samples_inputs.size() != all_samples_expected_outputs.size())
    {
        throw std::runtime_error("train() : nombre d’entrées différent du nombre de sorties");
    }

    int N = all_samples_inputs.size();

    for(size_t iter = 0 ; iter < (size_t)num_iter ; ++iter)
    {
        //choix aléatoire d'un échantillon
        int k = rand() % N;/////std::mt19937
        const std::vector<double>& inputs_k = all_samples_inputs[k];
        const std::vector<double>& expected_outputs_k = all_samples_expected_outputs[k];

        //propagation avant (mise à jour des Xlj)
        propagate(inputs_k, is_classification);

        //pour tous les neurones j de la dernière couche L on calcule deltas en sortie
        for(size_t j = 1 ; j <= (size_t)d[L] ; ++j)
        {
            double out = X[L][j];
            double error = out - expected_outputs_k[j - 1];

            if(is_classification)
            {
                //dérivée tanh
                error *= (1.0 - out * out);
            }

            deltas[L][j] = error;
        }

        //backprop sur les couches cachées
        for(int l = (int)L ; l >= 2 ; --l)
        {
            for(size_t i = 1 ; i <= (size_t)d[l - 1] ; ++i)
            {
                double total = 0.0;

                //somme des influences venant de la couche l
                for(size_t j = 1 ; j <= (size_t)d[l] ; ++j)
                {
                    total += W[l][i][j] * deltas[l][j];
                }

                //dérivée tanh de X[l-1][i]
                double act = X[l - 1][i];
                total *= (1.0 - act * act);

                deltas[l - 1][i] = total;
            }
        }

        //mise à jour des poids
        for(size_t l = 1 ; l <= (size_t)L ; ++l)
        {
            for(size_t i = 0 ; i <= (size_t)d[l - 1] ; ++i)//0 = biais
            {
                double Xi = X[l - 1][i];

                for(size_t j = 1 ; j <= (size_t)d[l] ; ++j)
                {
                    W[l][i][j] -= alpha * Xi * deltas[l][j];
                }
            }
        }
    }
}

//prédiction
std::vector<double> PMC::predict(const std::vector<double>& inputs, bool is_classification)
{
    //propagation avant
    propagate(inputs, is_classification);

    //retourne les neurones de la dernière couche (sans le biais j=0)
    std::vector<double> out;
    out.reserve(d[L]);//d[L] = nb neurones couche de sortie

    for (size_t j = 1; j <= (size_t)d[L]; ++j)
    {
        out.push_back(X[L][j]);
    }

    return out;
}

//extern "C"
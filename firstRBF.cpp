#if WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>

//================ RBF Network Class ===================
class RBF
{
public:
    //constructeur : n_hidden = nombre de RBF neurons, input_size, output_size
    RBF(int input_size, int n_hidden, int output_size)
        : input_size(input_size), n_hidden(n_hidden), output_size(output_size)
    {
        //initialisation des centres (c) et des sigma
        centers.resize(n_hidden, std::vector<double>(input_size, 0.0));
        sigmas.resize(n_hidden, 1.0); //valeur par défaut
        //initialisation des poids W (hidden -> output)
        W.resize(n_hidden + 1, std::vector<double>(output_size, 0.0)); //+1 pour biais
    }

    //calcule la distance euclidienne entre input et centre k
    double rbf_output(const std::vector<double>& input, int k)
    {
        double sum = 0.0;
        for(int i = 0; i < input_size; ++i)
        {
            double diff = input[i] - centers[k][i];
            sum += diff * diff;
        }
        return std::exp(-sum / (2 * sigmas[k] * sigmas[k]));
    }

    //propagation avant
    std::vector<double> predict(const std::vector<double>& input)
    {
        std::vector<double> hidden(n_hidden + 1, 1.0); // +1 pour biais
        for(int k = 0; k < n_hidden; ++k)
        {
            hidden[k] = rbf_output(input, k);
        }

        //sortie = somme hidden * poids
        std::vector<double> out(output_size, 0.0);
        for(int j = 0; j < output_size; ++j)
        {
            for(int k = 0; k <= n_hidden; ++k) //inclut biais
            {
                out[j] += W[k][j] * hidden[k];
            }
        }
        return out;
    }

    //entraînement simple pseudo-inverse (RBF typique)
    void train(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& Y)
    {
        int N = X.size();
        //fixer centres = premiers n_hidden samples
        for(int k = 0; k < n_hidden && k < N; ++k)
        {
            centers[k] = X[k];
        }

        //calculer hidden activations pour tous les samples
        std::vector<std::vector<double>> H(N, std::vector<double>(n_hidden + 1, 1.0)); //+1 pour biais
        for(int n = 0; n < N; ++n)
        {
            for(int k = 0; k < n_hidden; ++k)
            {
                H[n][k] = rbf_output(X[n], k);
            }
        }

        //pseudo-inverse H+ * Y pour obtenir W
        //pour garder le code simple ici on fera un apprentissage très basique (delta rule)
        double lr = 0.1;
        for(int iter = 0; iter < 1000; ++iter)
        {
            for(int n = 0; n < N; ++n)
            {
                std::vector<double> out = predict(X[n]);
                for(int j = 0; j < output_size; ++j)
                {
                    double error = out[j] - Y[n][j];
                    for(int k = 0; k <= n_hidden; ++k)
                    {
                        W[k][j] -= lr * error * H[n][k];
                    }
                }
            }
        }
    }

private:
    int input_size;
    int n_hidden;
    int output_size;
    std::vector<std::vector<double>> centers;
    std::vector<double> sigmas;
    std::vector<std::vector<double>> W;
};

//================ Exported C Functions ===================
extern "C"
{
    DLLEXPORT void* create_rbf(int input_size, int n_hidden, int output_size)
    {
        try
        {
            RBF* net = new RBF(input_size, n_hidden, output_size);
            return static_cast<void*>(net);
        }
        catch(...)
        {
            return nullptr;
        }
    }

    DLLEXPORT void destroy_rbf(void* handle)
    {
        if(handle) delete static_cast<RBF*>(handle);
    }

    DLLEXPORT int train_rbf(void* handle,
                             const double* X_flat,
                             const double* Y_flat,
                             int samples,
                             int input_size,
                             int output_size)
    {
        if(!handle || !X_flat || !Y_flat) return -1;
        RBF* net = static_cast<RBF*>(handle);
        std::vector<std::vector<double>> X(samples, std::vector<double>(input_size));
        std::vector<std::vector<double>> Y(samples, std::vector<double>(output_size));
        for(int n = 0; n < samples; ++n)
            for(int i = 0; i < input_size; ++i) X[n][i] = X_flat[n * input_size + i];
        for(int n = 0; n < samples; ++n)
            for(int j = 0; j < output_size; ++j) Y[n][j] = Y_flat[n * output_size + j];
        net->train(X, Y);
        return 0;
    }

    DLLEXPORT int predict_rbf(void* handle,
                              const double* input,
                              int input_size,
                              double* out_buffer,
                              int output_size)
    {
        if(!handle || !input || !out_buffer) return -1;
        RBF* net = static_cast<RBF*>(handle);
        std::vector<double> vin(input_size);
        for(int i = 0; i < input_size; ++i) vin[i] = input[i];
        std::vector<double> vout = net->predict(vin);
        if(vout.size() != (size_t)output_size) return -2;
        for(int j = 0; j < output_size; ++j) out_buffer[j] = vout[j];
        return 0;
    }
}


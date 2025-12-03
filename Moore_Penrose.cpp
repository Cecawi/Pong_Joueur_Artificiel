#include "Moore_Penrose.hpp"
#include <iostream>

//g++ -I "C:/Users/yecel/Downloads/eigen-5.0.0/eigen-5.0.0" -shared -o Moore_Penrose.dll Moore_Penrose.cpp

extern "C"
{
    //0 si succès, -1 si pointeurs null, -2 si dimensions invalides, -3 si pas assez d'échantillons, -4 si erreur d'entraînement
    __declspec(dllexport)
    int trainMoorePenrose(float* X, float* Y, int rows, int cols, float* outWeights, float* outBias)
    {
        if(!X || !Y || !outWeights || !outBias)
        {
            std::cerr << "Erreur : pointeurs null" << std::endl;
            return -1;
        }
        if(rows <= 0 || cols <= 0)
        {
            std::cerr << "Erreur : dimensions invalides (rows = " << rows << ", cols = " << cols << ")" << std::endl;
            return -2;
        }
        if(rows < cols + 1)//+1 pour le biais
        {
            std::cerr << "Erreur : pas assez d'échantillons (rows = " << rows << ", cols+1 = " << cols+1 << ")" << std::endl;
            return -3;
        }

        try
        {
            //conversion du tableau 1D C# en matrice Eigen
            Eigen::MatrixXf Xmat(rows, cols + 1);//+ 1 : colonne supplémentaire pour le biais
            Eigen::VectorXf Yvec(rows);

            for(int i = 0 ; i < rows ; ++i)
            {
                for(int j = 0 ; j < cols ; ++j)
                {
                    Xmat(i, j) = X[i * cols + j];
                }
                Xmat(i, cols) = 1.0f;//colonne de biais
                Yvec(i) = Y[i];
            }

            MoorePenrose model;
            //détection et correction de la colinéarité
            bool needUpdate = model.needUntrick(Xmat, rows, cols);

            if(needUpdate)
            {
                for(int i = 0 ; i < rows ; ++i)
                {
                    for(int j = 0 ; j < cols ; ++j)
                    {
                        X[i * cols + j] = Xmat(i, j);
                    }
                }
            }
            
            model.train(Xmat, Yvec);
            Eigen::VectorXf W = model.getWeights();

            //copie des poids dans le buffer de sortie pour Unity
            //copie les valeurs calculées par Eigen(W) dans un tableau simple (outWeights) que Unity envoie en paramètre
            for(int i = 0 ; i < cols ; ++i)
            {
                outWeights[i] = W(i);
            }
            *outBias = W(cols);

            return 0;//succès
        }
        catch(const std::exception& e)
        {
            std::cerr << "Erreur lors de l'entraînement : " << e.what() << std::endl;
            return -4;
        }
    }

    __declspec(dllexport)
    float predictMoorePenrose(float* weights, float bias, float* x, int size)
    {
        if(!weights || !x || size <= 0)
        {
            std::cerr << "Erreur : paramètres invalides pour predict" << std::endl;
            return 0.0f;
        }
        
        //création du vecteur de poids (avec le biais)
        Eigen::VectorXf W(size + 1);
        for(int i = 0 ; i < size ; ++i)
        {
            W(i) = weights[i];
        }
        W(size) = bias;//biais en dernière position

        //création du vecteur d'entrée (avec 1.0 pour le biais)
        Eigen::VectorXf inputVec(size + 1);
        for(int i = 0 ; i < size ; ++i)
        {
            inputVec(i) = x[i];
        }
        inputVec(size) = 1.0f;//1.0 pour le biais

        //utilisation de la classe MoorePenrose via setWeights et predict
        MoorePenrose model;
        model.setWeights(W);
        return model.predict(inputVec);
    }
}

void MoorePenrose::train(const Eigen::MatrixXf& X, const Eigen::VectorXf& y)
{
    weights = (X.transpose() * X).inverse() * X.transpose() * y;
}

float MoorePenrose::predict(const Eigen::VectorXf& x) const
{
    return weights.dot(x);//produit scalaire entre vecteur des poids (biais inclut) et vecteur x
}

//détecte s'il y a colinéarité et ajoute du bruit si besoin
bool MoorePenrose::needUntrick(Eigen::MatrixXf& Xmat, int rows, int cols)
{
    if(rows < 3)
	{
		return false;//pas assez de points pour détecter la colinéarité
	}
    
    //vérifier si les points sont colinéaires
    Eigen::MatrixXf XtX = Xmat.block(0, 0, rows, cols).transpose() * Xmat.block(0, 0, rows, cols);
    float det = XtX.determinant();
    
    if(-1e-6f < det && det < 1e-6f)//seuil de singularité
    {
        std::cout << "Points colinéaires détectés (det=" << det << "). Ajout de bruit" << std::endl;
        Xmat(rows - 1, 0) += 0.01f;
		return true;
    }

	return false;
}
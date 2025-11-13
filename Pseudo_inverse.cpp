
// #include <iostream>
// #include <vector>

// // ici c'est du 2x2 donc bon

// class LinearRegression {
// private:
//     float w1, w2, b; // coefficients
// public:
//     LinearRegression() : w1(0), w2(0), b(0) {}

//     void fit(const std::vector<std::vector<float>>& X, const std::vector<float>& Y) {
//         int n = X.size();

//         // Calcul des sommes nécessaires pour le pseudo-inverse (2D + biais)
//         float sumX1 = 0, sumX2 = 0, sumY = 0;
//         float sumX1X1 = 0, sumX2X2 = 0, sumX1X2 = 0;
//         float sumX1Y = 0, sumX2Y = 0;

//         for (int i = 0; i < n; ++i) {
//             float x1 = X[i][0];
//             float x2 = X[i][1];
//             float y  = Y[i];

//             sumX1 += x1;
//             sumX2 += x2;
//             sumY  += y;
//             sumX1X1 += x1 * x1;
//             sumX2X2 += x2 * x2;
//             sumX1X2 += x1 * x2;
//             sumX1Y  += x1 * y;
//             sumX2Y  += x2 * y;
//         }

//         //  calcule les coefficients de la matrice (X^T X) et son inverse à la main
//         float det = sumX1X1 * sumX2X2 - sumX1X2 * sumX1X2;
//         if (det == 0) {
//             std::cerr << "pas possible!" << std::endl;
//             return;
//         }

//         float inv11 =  sumX2X2 / det;
//         float inv12 = -sumX1X2 / det;
//         float inv21 = -sumX1X2 / det;
//         float inv22 =  sumX1X1 / det;

//         // On calcule W = (X^T X)^(-1) * X^T Y
//         w1 = inv11 * sumX1Y + inv12 * sumX2Y;
//         w2 = inv21 * sumX1Y + inv22 * sumX2Y;

//         // Biais = moyenne(Y - (w1*x1 + w2*x2))
//         float sumError = 0;
//         for (int i = 0; i < n; ++i)
//             sumError += (Y[i] - (w1 * X[i][0] + w2 * X[i][1]));
//         b = sumError / n;
//     }

//     float predict(float x1, float x2) const {
//         return w1 * x1 + w2 * x2 + b;
//     }

//     void printWeights() const {
//         std::cout << "w1 = " << w1 << ", w2 = " << w2 << ", bias = " << b << std::endl;
//     }
// };

// int main() {
//     std::vector<std::vector<float>> X = {
//         {1, 2}, {2, 3}, {3, 4}, {4, 5}
//     };
//     std::vector<float> Y = {3, 5, 7, 9};

//     LinearRegression model;
//     model.fit(X, Y);
//     model.printWeights();

//     std::cout << "Prediction (5,6)::: " << model.predict(5, 6) << std::endl;
// }





#include <iostream>
#include <vector>

class LinearRegression {
private:
    float w1, w2, b; // coefficients

public:
    LinearRegression() : w1(0), w2(0), b(0) {}

    void fit(const std::vector<std::vector<float>>& X, const std::vector<float>& Y) {
        int n = X.size();

        // Calcul des somme pour les 3x3 matrices
        float sum1 = n;
        float sumX1 = 0, sumX2 = 0;
        float sumX1X1 = 0, sumX2X2 = 0, sumX1X2 = 0;
        float sumY = 0, sumX1Y = 0, sumX2Y = 0;

        for (int i = 0; i < n; ++i) {
            float x1 = X[i][0];
            float x2 = X[i][1];
            float y = Y[i];

            sumX1 += x1;
            sumX2 += x2;
            sumY += y;
            sumX1X1 += x1 * x1;
            sumX2X2 += x2 * x2;
            sumX1X2 += x1 * x2;
            sumX1Y += x1 * y;
            sumX2Y += x2 * y;
        }

        // Matrice (X^T X)
        // [ sum1,  sumX1,  sumX2  ]
        // [ sumX1, sumX1X1, sumX1X2 ]
        // [ sumX2, sumX1X2, sumX2X2 ]
        float a = sum1, b_ = sumX1, c = sumX2;
        float d = sumX1, e = sumX1X1, f = sumX1X2;
        float g = sumX2, h = sumX1X2, i = sumX2X2;

        // Déterminant (pour inversion 3x3)
        float det = a*(e*i - f*h) - b_*(d*i - f*g) + c*(d*h - e*g);

        if (det == 0) {
            std::cerr << "pas possible!" << std::endl;
            return;
        }

        // Inverse de (X^T X)
        float inv[3][3];
        inv[0][0] =  (e*i - f*h) / det;
        inv[0][1] = -(b_*i - c*h) / det;
        inv[0][2] =  (b_*f - c*e) / det;
        inv[1][0] = -(d*i - f*g) / det;
        inv[1][1] =  (a*i - c*g) / det;
        inv[1][2] = -(a*f - c*d) / det;
        inv[2][0] =  (d*h - e*g) / det;
        inv[2][1] = -(a*h - b_*g) / det;
        inv[2][2] =  (a*e - b_*d) / det;

        // Matrice (X^T Y)
        float XtY[3] = { sumY, sumX1Y, sumX2Y };

        // W = (X^T X)^(-1) * (X^T Y)
        float result[3] = {0};
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                result[r] += inv[r][c] * XtY[c];

        b = result[0];  // biais
        w1 = result[1];
        w2 = result[2];
    }

    float predict(float x1, float x2) const {
        return b + w1 * x1 + w2 * x2;
    }

    void printWeights() const {
        std::cout << "b = " << b << ", w1 = " << w1 << ", w2 = " << w2 << std::endl;
    }
};

int main() {
    // Données d'exemple (avec biais implicite)
    std::vector<std::vector<float>> X = {
    {1, 2}, {2, 5}, {3, 7}, {4, 8}
};
    std::vector<float> Y = {4, 9, 13, 16};

    LinearRegression model;
    model.fit(X, Y);
    model.printWeights();

    std::cout << "Prediction (5,6): " << model.predict(3, 6) << std::endl;
}

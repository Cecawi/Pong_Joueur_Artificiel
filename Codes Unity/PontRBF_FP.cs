using System;
using System.Runtime.InteropServices;
using UnityEngine;
using System.Collections.Generic;
using System.Linq;

public class PontRBF_FP : MonoBehaviour
{
    // =======================================================
    // --- VARIABLES MEMBRES & IMPORTS DLL (CHAMPS DE CLASSE) ---
    // =======================================================

    // Ces variables sont accessibles PARTOUT dans cette classe.
    private float posX = 0f;
    private float posY = 30f;
    private float decalageX = 30f;
    private float decalageY = -6f;
    private float tailleAxesRepere = 5f;

    [DllImport("RBF_FP", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr create_rbf(int[] sizes, int layers_count);

    [DllImport("RBF_FP", CallingConvention = CallingConvention.Cdecl)]
    private static extern void destroy_rbf(IntPtr handle);

    [DllImport("RBF_FP", CallingConvention = CallingConvention.Cdecl)]
    private static extern int train_rbf
    (
        IntPtr handle, double[] X_flat, double[] Y_flat, int samples, int input_size,
        int output_size, int is_classification, int num_iter, double alpha
    );

    [DllImport("RBF_FP", CallingConvention = CallingConvention.Cdecl)]
    private static extern int predict_rbf
    (
        IntPtr handle, double[] input, int input_size,
        double[] out_buffer, int output_size, int is_classification
    );

    // =======================================================
    // --- METHODE START (Les Tests) ---
    // =======================================================

    void Start()
    {
        // On utilise les champs de classe directement pour le suivi de position.
        float currentPosX = posX;
        float currentPosY = posY;
        System.Random rnd = new System.Random();

        // -------------------------------------------------------
        // # 1. CLASSIFICATION LINÉAIRE SIMPLE 
        // -------------------------------------------------------
        string titre1 = "RBF_FP Classification Linéaire Simple";
        AfficherTitre(titre1, currentPosX + 2.5f, currentPosY + 1.5f);

        double[] xLineaireSimple = { 1, 1, 2, 3, 3, 3 };
        double[] yLineaireSimple = { 1, -1, -1 };

        // APPEL CORRIGÉ (Ligne 60 approx.) : Moins d'arguments nécessaires
        TestsRBFClassification(
            new int[] { 2, 2, 1 }, 3, xLineaireSimple, yLineaireSimple,
            200, 0f, 4f, 0f, 4f,
            currentPosX, currentPosY, "Linéaire Simple"
        );
        currentPosX += decalageX;

        // -------------------------------------------------------
        // # 2. CLASSIFICATION LINÉAIRE MULTIPLE 
        // -------------------------------------------------------
        string titre2 = "RBF_FP Classification Linéaire Multiple";
        AfficherTitre(titre2, currentPosX + 2.5f, currentPosY + 1.5f);

        double[] xLineaireMultiple = GenereDonneesLineaires(50, 1.0, 1.0, 2.5, 2.5, rnd);
        double[] yLineaireMultiple = GenereLabelsLineaires(50);

        // APPEL CORRIGÉ (Ligne 76 approx.)
        TestsRBFClassification(
            new int[] { 2, 3, 1 }, 3, xLineaireMultiple, yLineaireMultiple,
            200, 0f, 4f, 0f, 4f,
            currentPosX, currentPosY, "Linéaire Multiple"
        );
        currentPosX += decalageX;

        // -------------------------------------------------------
        // # 3. CLASSIFICATION XOR (Non Linéaire Classique)
        // -------------------------------------------------------
        string titre3 = "RBF_FP XOR (Non Linéaire Classique)";
        AfficherTitre(titre3, currentPosX + 0.5f, currentPosY + 1.5f);

        double[] xXor = { 1, 0, 0, 1, 0, 0, 1, 1 };
        double[] yXor = { 1, 1, -1, -1 };

        // APPEL CORRIGÉ (Ligne 93 approx.)
        TestsRBFClassification(
            new int[] { 2, 4, 1 }, 3, xXor, yXor,
            200, -0.5f, 1.5f, -0.5f, 1.5f,
            currentPosX, currentPosY, "XOR"
        );
        currentPosX += decalageX;

        // -------------------------------------------------------
        // # 4. CLASSIFICATION CROSS (Non Linéaire Tricky)
        // -------------------------------------------------------
        currentPosX = posX;
        currentPosY += decalageY;

        string titre4 = "RBF_FP Classification Croix (Non Linéaire)";
        AfficherTitre(titre4, currentPosX + 1.5f, currentPosY + 1.5f);

        double[] xCroix = new double[1000];
        double[] yCroix = new double[500];

        for (int i = 0; i < 500; i++)
        {
            double px = rnd.NextDouble() * 2.0 - 1.0;
            double py = rnd.NextDouble() * 2.0 - 1.0;

            xCroix[2 * i] = px;
            xCroix[2 * i + 1] = py;

            yCroix[i] = (Math.Abs(px) <= 0.3 || Math.Abs(py) <= 0.3) ? 1.0 : -1.0;
        }

        // APPEL CORRIGÉ (Ligne 124 approx.)
        TestsRBFClassification(
            new int[] { 2, 15, 1 }, 3, xCroix, yCroix,
            500, -1.5f, 1.5f, -1.5f, 1.5f,
            currentPosX, currentPosY, "Croix (15 C.)"
        );
        currentPosX += decalageX;

        // -------------------------------------------------------
        // # 5. RÉGRESSION LINÉAIRE SIMPLE 2D 
        // -------------------------------------------------------
        string titre5 = "RBF_FP Régression Linéaire Simple (1D)";
        AfficherTitre(titre5, currentPosX + 2.5f, currentPosY + 1.5f);

        double[] xRegSimple = { 1, 2, 3, 4 };
        double[] yRegSimple = { 2.2, 3.1, 4.3, 5.4 };

        // APPEL CORRIGÉ (Ligne 141 approx.)
        TestsRBFRegression(
            new int[] { 1, 1, 1 }, 3, xRegSimple, yRegSimple,
            0f, 6f, 0f, 6f,
            currentPosX, currentPosY, "Reg. Lin. 1D"
        );
        currentPosX += decalageX;

        // -------------------------------------------------------
        // # 6. RÉGRESSION NON LINÉAIRE SIMPLE 3D
        // -------------------------------------------------------
        string titre6 = "RBF_FP Régression Non Linéaire 3D";
        AfficherTitre(titre6, currentPosX + 2.5f, currentPosY + 1.5f);

        double[] xReg3D = { 1, 0, 0, 1, 1, 1, 0, 0 };
        double[] yReg3D = { 2, 1, -2, -1 };

        // APPEL CORRIGÉ (Ligne 158 approx.)
        TestsRBFRegression3D(
            new int[] { 2, 4, 1 }, 3, xReg3D, yReg3D,
            0f, 1f, 0f, 1f,
            currentPosX, currentPosY, "Reg. Non Lin. 3D"
        );
    }

    // =======================================================
    // --- NOUVELLES SIGNATURES DE METHODES (SANS DECALAGEX/Y REDONDANTS) ---
    // =======================================================

    // SIGNATURE CORRIGÉE (Remplacement des arguments redondants par des champs de classe)
    void TestsRBFClassification
    (
        int[] RbfSizes, int TailleRbf, double[] DoneesX, double[] DoneesY,
        int NbrXAPred, float Xg, float Xd, float Yg, float Yd,
        float PosX, float PosY, string testName // Suppression des ref float, DecalageX/Y, TailleAxesRepere
    )
    {
        double[] X_flat = DoneesX;
        int nbrPoints = DoneesX.Length / 2;
        int sortieParPoint = DoneesY.Length / nbrPoints;
        double[] Y_flat = DoneesY;

        Vector2[] donneesAleaAPredire = GenerePtsAlea(NbrXAPred, Xg, Xd, Yg, Yd);

        // Affiche 3 versions du résultat optimal (pour la visibilité)
        for (int affichage = 0; affichage < 3; affichage++)
        {
            IntPtr ptrRbf = create_rbf(RbfSizes, TailleRbf);
            try
            {
                train_rbf(ptrRbf, X_flat, Y_flat, nbrPoints, RbfSizes[0], RbfSizes[RbfSizes.Length - 1], 1, 0, 0.0);

                // Utilisation du champ de classe 'decalageX'
                float currentPosX = PosX + affichage * decalageX / 3f;
                AfficherRBFClassification(
                    ptrRbf, X_flat, Y_flat, currentPosX, PosY, tailleAxesRepere, // Utilisation du champ 'tailleAxesRepere'
                    donneesAleaAPredire, sortieParPoint, $"{testName} ({affichage + 1})"
                );
            }
            finally
            {
                destroy_rbf(ptrRbf);
            }
        }
    }

    // SIGNATURE CORRIGÉE
    void TestsRBFRegression
    (
        int[] RbfSizes, int TailleRbf, double[] DoneesX, double[] DoneesY,
        float Xg, float Xd, float Yg, float Yd,
        float PosX, float PosY, string testName
    )
    {
        double[] X_flat = DoneesX;
        double[] Y_flat = DoneesY;
        int nbrPoints = DoneesX.Length;

        List<double> predictionInputs = new List<double>();
        for (double x = Xg; x <= Xd; x += (Xd - Xg) / 100.0)
        {
            predictionInputs.Add(x);
        }

        IntPtr ptrRbf = create_rbf(RbfSizes, TailleRbf);
        try
        {
            train_rbf(ptrRbf, X_flat, Y_flat, nbrPoints, RbfSizes[0], RbfSizes[RbfSizes.Length - 1], 0, 0, 0.0);

            AfficherRBFRegression(
                ptrRbf, X_flat, Y_flat, PosX, PosY, tailleAxesRepere, // Utilisation du champ 'tailleAxesRepere'
                predictionInputs.ToArray(), Xg, Xd, Yg, Yd, testName
            );
        }
        finally
        {
            destroy_rbf(ptrRbf);
        }
    }

    // SIGNATURE CORRIGÉE
    void TestsRBFRegression3D
    (
        int[] RbfSizes, int TailleRbf, double[] DoneesX, double[] DoneesY,
        float Xg, float Xd, float Yg, float Yd,
        float PosX, float PosY, string testName
    )
    {
        int nbrPoints = DoneesY.Length;

        IntPtr ptrRbf = create_rbf(RbfSizes, TailleRbf);

        try
        {
            train_rbf(ptrRbf, DoneesX, DoneesY, nbrPoints, 2, 1, 0, 0, 0.0);

            AfficherRegression3D(
                ptrRbf, DoneesX, DoneesY,
                PosX, PosY, tailleAxesRepere, // Utilisation du champ 'tailleAxesRepere'
                Xg, Xd, Yg, Yd, testName
            );
        }
        finally
        {
            destroy_rbf(ptrRbf);
        }
    }

    // =======================================================
    // --- AUTRES FONCTIONS (UTILISANT LES CHAMPS DE CLASSE) ---
    // =======================================================

    // ... (Reste des fonctions utilitaires, elles doivent utiliser les champs de classe (posX, decalageX, etc.) directement, sans les recevoir en argument, sauf pour les positions de base PosX/PosY) ...

    // Correction de la taille du tableau X
    double[] GenereDonneesLineaires(int count, double x1, double y1, double x2, double y2, System.Random rnd)
    {
        // Taille corrigée : 2 classes * count points * 2 dimensions
        double[] X = new double[count * 4];

        // Première classe (0 à 2*count - 1)
        for (int i = 0; i < count; i++)
        {
            X[i * 2] = rnd.NextDouble() * 0.9 + x1;
            X[i * 2 + 1] = rnd.NextDouble() * 0.9 + y1;
        }

        // Deuxième classe (démarrage à l'indice 2*count)
        for (int i = 0; i < count; i++)
        {
            // L'indice commence maintenant à 2 * count
            X[(count * 2) + (i * 2)] = rnd.NextDouble() * 0.9 + x2;
            X[(count * 2) + (i * 2) + 1] = rnd.NextDouble() * 0.9 + y2;
        }
        return X;
    }

    double[] GenereLabelsLineaires(int count)
    {
        double[] Y = new double[count * 2];
        for (int i = 0; i < count; i++) Y[i] = 1.0;
        for (int i = count; i < count * 2; i++) Y[i] = -1.0;
        return Y;
    }

    // UTILS POUR AFFICHAGE (A implémenter dans projet Unity)
    void AfficherTitre(string text, float PosX, float PosY)
    {
        GameObject textObject = new GameObject("Test Title " + text);
        TextMesh textMesh = textObject.AddComponent<TextMesh>();

        textMesh.text = text;
        textMesh.fontSize = 25;
        textMesh.color = Color.black;

        textObject.transform.position = new Vector3(PosX, PosY, -1.0f);
        textObject.transform.localScale = new Vector3(0.15f, 0.15f, 0.15f);
    }

    void CreerAxe(float PosX, float PosY, Color col, Vector3 A, Vector3 B, string name)
    {
        GameObject axis = new GameObject(name);
        var lr = axis.AddComponent<LineRenderer>();
        lr.positionCount = 2;
        lr.startWidth = lr.endWidth = 0.05f;
        lr.material = new Material(Shader.Find("Sprites/Default"));
        lr.startColor = lr.endColor = col;
        lr.SetPosition(0, A);
        lr.SetPosition(1, B);
    }

    void AfficherRepere(float PosX, float PosY, float Taille, string title)
    {
        CreerAxe(PosX, PosY, Color.red, new Vector3(0 + PosX, PosY, 0), new Vector3(Taille + PosX, PosY, 0), $"Axe x {title}");
        CreerAxe(PosX, PosY, Color.green, new Vector3(PosX, 0 + PosY, 0), new Vector3(PosX, Taille + PosY, 0), $"Axe y {title}");
        CreerAxe(PosX, PosY, Color.blue, new Vector3(PosX, PosY, 0), new Vector3(PosX, PosY, Taille), $"Axe z {title}");
    }

    Vector2[] GenerePtsAlea(int NbrXAPred, float Xg, float Xd, float Yg, float Yd)
    {
        Vector2[] pts = new Vector2[NbrXAPred];
        for (int i = 0; i < NbrXAPred; i++)
        {
            pts[i] = new Vector2(UnityEngine.Random.Range(Xg, Xd), UnityEngine.Random.Range(Yg, Yd));
        }
        return pts;
    }

    // AfficherRBFClassification (Utilise TailleAxesRepere du champ de classe)
    void AfficherRBFClassification
    (
        IntPtr PtrRbf, double[] DoneesX, double[] DoneesY,
        float PosX, float PosY, float TailleAxesRepere,
        Vector2[] donneesAleaAPredire, int sortieParPoint, string title
    )
    {
        AfficherRepere(PosX, PosY, TailleAxesRepere, title);
        int nbrPoints = DoneesX.Length / 2;

        // Points d'entraînement
        for (int i = 0; i < nbrPoints; i++)
        {
            Color expectedCol = DoneesY[i] > 0 ? Color.blue : Color.red;

            var sph = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sph.GetComponent<Renderer>().material.color = expectedCol;
            sph.transform.position = new Vector3((float)DoneesX[2 * i] + PosX, (float)DoneesX[2 * i + 1] + PosY, 0);
            sph.transform.localScale = Vector3.one * 0.15f;
        }

        // Prédictions (zone de décision)
        foreach (var pt in donneesAleaAPredire)
        {
            double[] inp = { pt.x, pt.y };
            double[] outp = new double[sortieParPoint];

            predict_rbf(PtrRbf, inp, 2, outp, sortieParPoint, 1);

            Color predictedCol = outp[0] > 0 ? Color.green : Color.yellow;

            var sph = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sph.GetComponent<Renderer>().material.color = predictedCol;
            sph.transform.position = new Vector3(pt.x + PosX, pt.y + PosY, 0);
            sph.transform.localScale = Vector3.one * 0.1f;
        }
    }

    // AfficherRBFRegression (Utilise TailleAxesRepere du champ de classe)
    void AfficherRBFRegression
    (
        IntPtr PtrRbf, double[] DoneesX, double[] DoneesY,
        float PosX, float PosY, float TailleAxesRepere,
        double[] predictionInputs, float Xg, float Xd, float Yg, float Yd, string title
    )
    {
        AfficherRepere(PosX, PosY, TailleAxesRepere, title);

        for (int i = 0; i < DoneesY.Length; i++)
        {
            var sph = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sph.GetComponent<Renderer>().material.color = Color.blue;
            sph.transform.position = new Vector3((float)DoneesX[i] + PosX, (float)DoneesY[i] + PosY, 0);
            sph.transform.localScale = Vector3.one * 0.15f;
        }

        GameObject courbe = new GameObject("CourbeRegressionRBF");
        var lr = courbe.AddComponent<LineRenderer>();
        lr.positionCount = predictionInputs.Length;
        lr.startWidth = lr.endWidth = 0.05f;
        lr.material = new Material(Shader.Find("Sprites/Default"));
        lr.startColor = lr.endColor = Color.red;

        for (int i = 0; i < predictionInputs.Length; i++)
        {
            double x = predictionInputs[i];
            double[] inp = { x };
            double[] outp = new double[1];
            predict_rbf(PtrRbf, inp, 1, outp, 1, 0);

            var sph = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sph.GetComponent<Renderer>().material.color = Color.green;
            sph.transform.position = new Vector3((float)x + PosX, (float)outp[0] + PosY, 0);
            sph.transform.localScale = Vector3.one * 0.08f;

            lr.SetPosition(i, new Vector3((float)x + PosX, (float)outp[0] + PosY, 0));
        }
    }

    // AfficherRegression3D (Utilise TailleAxesRepere du champ de classe)
    void AfficherRegression3D
    (
        IntPtr PtrRbf,
        double[] DoneesX, double[] DoneesY,
        float PosX, float PosY, float TailleAxesRepere,
        float Xg, float Xd, float Yg, float Yd, string title
    )
    {
        AfficherRepere(PosX, PosY, TailleAxesRepere, title);

        int nbrPoints = DoneesY.Length;

        // Points d'entraînement
        for (int i = 0; i < nbrPoints; i++)
        {
            float x = (float)DoneesX[2 * i];
            float y = (float)DoneesX[2 * i + 1];
            float z = (float)DoneesY[i];

            var sph = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sph.GetComponent<Renderer>().material.color = Color.blue;
            sph.transform.position = new Vector3(x + PosX, y + PosY, z);
            sph.transform.localScale = Vector3.one * 0.15f;
        }

        // Surface prédite → grille 20 x 20
        int N = 20;
        for (int ix = 0; ix < N; ix++)
        {
            for (int iy = 0; iy < N; iy++)
            {
                float x = Mathf.Lerp(Xg, Xd, ix / (float)(N - 1));
                float y = Mathf.Lerp(Yg, Yd, iy / (float)(N - 1));

                double[] inp = { x, y };
                double[] outp = new double[1];
                predict_rbf(PtrRbf, inp, 2, outp, 1, 0);

                var sph = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                sph.GetComponent<Renderer>().material.color = Color.green;
                sph.transform.position = new Vector3(x + PosX, y + PosY, (float)outp[0]);
                sph.transform.localScale = Vector3.one * 0.05f;
            }
        }
    }
}
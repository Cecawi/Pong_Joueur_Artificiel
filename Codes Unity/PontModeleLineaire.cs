using System;
using System.Runtime.InteropServices;
using UnityEngine;

public class PontModeleLineaire : MonoBehaviour
{
    [DllImport("Modele_lineaire.dll", CallingConvention = CallingConvention.Cdecl)]
    private static extern void trainLinearModel(float[] X, float[] y, int rows, int cols, int epochs, float lr, float[] weights, ref float bias);

    void Start()
    {
        float xOffset = 0f;//decalage sur X entre chaque cas
        float stepOffset = 10f;//espacement entre les cas

//////////notre test : y = 2x + 1
        float[] X1 = { 1f, 2f, 3f, 4f };
        float[] Y1 = { 3f, 5f, 7f, 9f };
        RunTest2D(X1, Y1, 0.01f, 1000, xOffset);
        xOffset += stepOffset;

//////////lineaire simple 2D : y = x + 1/////modifié lors de la présentation de la 1ere etape intermediaire
        float[] X2 = { 1f, 2f };
        float[] Y2 = { 2f, 1f };
        RunTest2D(X2, Y2, 0.01f, 100000, xOffset);
        xOffset += stepOffset;

//////////non lineaire simple 2D : y ~= -0.75 * x^2 + 3.25 * x - 0.5
        float[] XnonLin2D = { 1f, 2f, 3f };
        float[] YnonLin2D = { 2f, 3f, 2.5f };
        RunTest2D(XnonLin2D, YnonLin2D, 0.01f, 1000, xOffset);
        xOffset += stepOffset;

//////////lineaire simple 3D : y = w1*x1 + w2*x2 + b
        float[,] X3D = { {1,1}, {2,2}, {3,1} };
        float[] Y3D = { 2f, 3f, 2.5f };
        RunTest3D(X3D, Y3D, 0.01f, 1000, xOffset);
        xOffset += stepOffset;

//////////lineaire tricky 3D : x1 et x2 evoluent ensemble
        float[,] Xtricky = { {1,1}, {2,2}, {3,3} };
        float[] Ytricky = { 1f, 2f, 3f };
        RunTest3D(Xtricky, Ytricky, 0.01f, 1000, xOffset);
        xOffset += stepOffset;

//////////non lineaire simple 3D : XOR-like
        float[,] XnonLin3D = { {1,0}, {0,1}, {1,1}, {0,0} };
        float[] YnonLin3D = { 2f, 1f, -2f, -1f };
        RunTest3D(XnonLin3D, YnonLin3D, 0.01f, 1000, xOffset);
    }

    void RunTest2D(float[] Xdata, float[] yData, float lr, int epochs, float xOffset)
    {
        int rows = Xdata.Length;
        int cols = 1;
        float[] w = new float[cols];
        float b = 0f;
        float yOffset = 15f;

        trainLinearModel(Xdata, yData, rows, cols, epochs, lr, w, ref b);
        Debug.Log($"2D Test : Poids appris : {w[0]}, biais : {b}");

        //points bleus
        for (int i = 0; i < rows; i++)
        {
            var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.transform.position = new Vector3(Xdata[i] + xOffset, yData[i] + yOffset, 0);
            sphere.transform.localScale = Vector3.one * 0.2f;
            sphere.GetComponent<Renderer>().material.color = Color.blue;
        }

        //droite rouge
        GameObject lineObj = new GameObject("RegressionLine1D");
        var line = lineObj.AddComponent<LineRenderer>();
        line.positionCount = 2;
        line.SetPosition(0, new Vector3(0 + xOffset, b + yOffset, 0));
        line.SetPosition(1, new Vector3(5 + xOffset, w[0]*5 + b + yOffset, 0));
        line.startWidth = line.endWidth = 0.05f;
        line.material = new Material(Shader.Find("Sprites/Default"));
        line.startColor = line.endColor = Color.red;
    }

    void RunTest3D(float[,] Xdata, float[] yData, float lr, int epochs, float xOffset)
    {
        int rows = yData.Length;
        int cols = Xdata.GetLength(1);
        float[] w = new float[cols];
        float b = 0f;
        float yOffset = 15f;

        //conversion pour la DLL si necessaire
        float[] Xflat = new float[rows*cols];
        for(int i=0; i<rows; i++)
            for(int j=0; j<cols; j++)
                Xflat[i*cols + j] = Xdata[i,j];

        trainLinearModel(Xflat, yData, rows, cols, epochs, lr, w, ref b);
        Debug.Log($"3D Test : Poids appris : {string.Join(", ", w)}, biais : {b}");

        //points bleus
        for(int i=0; i<rows; i++)
        {
            var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            float x1 = Xdata[i,0];
            float x2 = (cols>1)? Xdata[i,1] : 0f;
            float y = yData[i];
            sphere.transform.position = new Vector3(x1 + xOffset, y + yOffset, x2);
            sphere.transform.localScale = Vector3.one * 0.2f;
            sphere.GetComponent<Renderer>().material.color = Color.blue;
        }

        //plan rouge (approximation lineaire)
        GameObject planeObj = new GameObject("RegressionPlane2D");
        var mesh = new Mesh();
        planeObj.AddComponent<MeshFilter>().mesh = mesh;
        planeObj.AddComponent<MeshRenderer>().material = new Material(Shader.Find("Standard"));
        planeObj.GetComponent<Renderer>().material.color = Color.red;

        Vector3[] vertices = new Vector3[4];
        float size = 5f;
        if(cols == 1)//droite 2D
        {
            vertices[0] = new Vector3(0 + xOffset, b + yOffset, 0);
            vertices[1] = new Vector3(size + xOffset, w[0]*size + b + yOffset, 0);
            vertices[2] = new Vector3(0 + xOffset, b + yOffset, 0.01f);
            vertices[3] = new Vector3(size + xOffset, w[0]*size + b + yOffset, 0.01f);
        }
        else//plan 3D
        {
            float w1 = w[0];
            float w2 = w[1];
            vertices[0] = new Vector3(0 + xOffset, b + yOffset, 0);
            vertices[1] = new Vector3(size + xOffset, w1*size + b + yOffset, 0);
            vertices[2] = new Vector3(0 + xOffset, w2*size + b + yOffset, size);
            vertices[3] = new Vector3(size + xOffset, w1*size + w2*size + b + yOffset, size);
        }
        int[] tris = {0,2,1, 2,3,1};
        mesh.vertices = vertices;
        mesh.triangles = tris;
        mesh.RecalculateNormals();
    }
}
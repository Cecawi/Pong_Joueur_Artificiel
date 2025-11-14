using System;
using System.Runtime.InteropServices;
using UnityEngine;

public class PontMoorePenrose : MonoBehaviour
{
    [DllImport("Moore_Penrose")]
    private static extern void trainMoorePenrose(
        float[] X, float[] Y, int rows, int cols,
        float[] outWeights, ref float outBias
    );

    void Start()
    {
        float xOffset = 0f;
        float stepOffset = 10f;

//////////notre test : y = 2x + 1
        float[] X1 = { 1f, 2f, 3f, 4f };
        float[] Y1 = { 3f, 5f, 7f, 9f };
        RunTest2D(X1, Y1, xOffset);
        xOffset += stepOffset;

//////////lineaire simple 2D : y = x + 1
        float[] X2 = { 1f, 2f };
        float[] Y2 = { 2f, 3f };
        RunTest2D(X2, Y2, xOffset);
        xOffset += stepOffset;

//////////non lineaire simple 2D : y ~= -0.75 * x^2 + 3.25 * x - 0.5
        float[] X3 = { 1f, 2f, 3f };
        float[] Y3 = { 2f, 3f, 2.5f };
        RunTest2D(X3, Y3, xOffset);
        xOffset += stepOffset;

//////////lineaire simple 3D : y = w1*x1 + w2*x2 + b
        float[,] X3D = { {1,1}, {2,2}, {3,1} };
        float[] Y3D = { 2f, 3f, 2.5f };
        RunTest3D(X3D, Y3D, xOffset);
        xOffset += stepOffset;

//////////lineaire tricky 3D : x1 et x2 evoluent ensemble
        float[,] Xt = { {1,1}, {2,2}, {3,3} };
        float[] Yt = { 1f, 2f, 3f };
        RunTest3D(Xt, Yt, xOffset);
        xOffset += stepOffset;

//////////lineaire tricky 3D : légèrement modifié
        float[,] Xtm = { {1,1}, {2,2.01f}, {3,3} };
        float[] Ytm = { 1f, 2f, 3f };
        RunTest3D(Xtm, Ytm, xOffset);
        xOffset += stepOffset;

//////////non lineaire simple 3D : XOR-like
        float[,] Xnl = { {1,0}, {0,1}, {1,1}, {0,0} };
        float[] Ynl = { 2f, 1f, -2f, -1f };
        RunTest3D(Xnl, Ynl, xOffset);
    }

    void RunTest2D(float[] Xdata, float[] yData, float xOffset)
    {
        int rows = Xdata.Length;
        int cols = 1;

        float[] w = new float[cols];
        float b = 0f;

        trainMoorePenrose(Xdata, yData, rows, cols, w, ref b);
        Debug.Log($"2D Test => w = {w[0]}, b = {b}");

        //Points
        for (int i = 0; i < rows; i++)
        {
            var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            sphere.transform.position = new Vector3(Xdata[i] + xOffset, yData[i], 0);
            sphere.transform.localScale = Vector3.one * 0.2f;
            sphere.GetComponent<Renderer>().material.color = Color.blue;
        }

        //Droite
        GameObject lineObj = new GameObject("RegressionLine1D");
        var line = lineObj.AddComponent<LineRenderer>();
        line.positionCount = 2;
        line.SetPosition(0, new Vector3(0 + xOffset, b, 0));
        line.SetPosition(1, new Vector3(5 + xOffset, w[0] * 5 + b, 0));
        line.startWidth = line.endWidth = 0.05f;
        line.material = new Material(Shader.Find("Sprites/Default"));
        line.startColor = line.endColor = Color.red;
    }

    void RunTest3D(float[,] Xdata, float[] yData, float xOffset)
    {
        int rows = yData.Length;
        int cols = Xdata.GetLength(1);

        float[] w = new float[cols];
        float b = 0f;

        float[] Xflat = new float[rows * cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                Xflat[i * cols + j] = Xdata[i, j];

        trainMoorePenrose(Xflat, yData, rows, cols, w, ref b);
        Debug.Log($"3D Test => w = {string.Join(", ", w)}, b = {b}");

        //Points
        for (int i = 0; i < rows; i++)
        {
            var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            float x1 = Xdata[i, 0];
            float x2 = (cols > 1 ? Xdata[i, 1] : 0f);
            float y = yData[i];
            sphere.transform.position = new Vector3(x1 + xOffset, y, x2);
            sphere.transform.localScale = Vector3.one * 0.2f;
            sphere.GetComponent<Renderer>().material.color = Color.blue;
        }

        //Plan
        GameObject planeObj = new GameObject("RegressionPlane");
        var mesh = new Mesh();
        planeObj.AddComponent<MeshFilter>().mesh = mesh;
        planeObj.AddComponent<MeshRenderer>().material = new Material(Shader.Find("Standard"));
        planeObj.GetComponent<Renderer>().material.color = Color.red;

        Vector3[] vertices = new Vector3[4];
        float size = 5f;

        if (cols == 1)
        {
            vertices[0] = new Vector3(0 + xOffset, b, 0);
            vertices[1] = new Vector3(size + xOffset, w[0] * size + b, 0);
            vertices[2] = new Vector3(0 + xOffset, b, 0.01f);
            vertices[3] = new Vector3(size + xOffset, w[0] * size + b, 0.01f);
        }
        else
        {
            float w1 = w[0];
            float w2 = w[1];
            vertices[0] = new Vector3(0 + xOffset, b, 0);
            vertices[1] = new Vector3(size + xOffset, w1 * size + b, 0);
            vertices[2] = new Vector3(0 + xOffset, w2 * size + b, size);
            vertices[3] = new Vector3(size + xOffset, w1 * size + w2 * size + b, size);
        }

        mesh.vertices = vertices;
        mesh.triangles = new int[] { 0, 2, 1, 2, 3, 1 };
        mesh.RecalculateNormals();
    }
}
using System.Text.Json;
using static System.Math;

namespace RetrievalSystem;


public record Bm25Param(double K1, double B, double K3)
{
    public static Bm25Param Default { get; } = new(1.5, 0.75, 1.5);
}

public record Bm25Model(Dictionary<string, Dictionary<string, double>> Tf, Dictionary<string, double> Idf, Dictionary<string, int> DocLens,
    double AvgDocLen, Bm25Param Param);
public class Bm25
{
    private readonly Dictionary<string, double> _idf = new();
    private readonly Dictionary<string, Dictionary<string, double>> _tf = new();
    private Dictionary<string, int> _docLens = new();
    private double _avgDocLen;

    public static Bm25 Load(string path)
    {
        return new(JsonSerializer.Deserialize<Bm25Model>(
            File.ReadAllText(path))!);
    }

    public static Bm25 Up(string cachePath, Func<Dictionary<string, string[]>> getDocs)
    {
        if (File.Exists(cachePath))
        {
            return Bm25.Load(cachePath);
        }

        var model = new Bm25(Bm25Param.Default);
        model.Fit(getDocs());
        model.Save(cachePath);
        return model;
    }
    public void Save(string path)
    {
        var model = new Bm25Model(
            _tf, _idf, _docLens, _avgDocLen, Param
        );
        if (File.Exists(path)) File.Delete(path);
        File.WriteAllText(path, JsonSerializer.Serialize(model));
    }
    private Bm25(Bm25Model model) : this(model.Param)
        => (_tf, _idf, _docLens, _avgDocLen, _) = model;

    public Bm25(Bm25Param param)
    {
        Param = param;
    }

    private Bm25Param Param { get; }

    public void Fit(Dictionary<string, string[]> docs)
    {
        this._docLens = docs.ToDictionary(kv => kv.Key, kv => kv.Value.Length);
        this._avgDocLen = _docLens.Values.Average();
        var df = new Dictionary<string, double>();
        foreach (var (key, doc) in docs)
        {
            var tf = new Dictionary<string, double>();
            foreach (var word in doc)
            {
                if (!tf.ContainsKey(word))
                {
                    tf.Add(word, 1);
                    df[word] =
                        df.TryGetValue(word, out var v) ?
                        v + 1 : 1;
                }
                else
                {
                    tf[word]++;
                }
            }

            _tf.Add(key, tf);
            foreach (var (word, docFreq) in df)
            {
                _idf[word] = Log(docs.Count / docFreq);
            }
        }
    }

    public (string, double)[] ScoreOf(string[] queryWords)
    {
        var (k1, b, k3) = Param;
        var results = new (string, double)[_tf.Count];
        var queryTf = new Dictionary<string, int>();
        foreach (var word in queryWords)
        {
            queryTf[word] = queryTf.TryGetValue(word, out var v) ? v + 1 : 1;

        }

        var i = 0;
        foreach (var (doc, tf) in _tf)
        {
            results[i] = (doc, queryTf.Where(q => tf.ContainsKey(q.Key))
                .Sum(q =>
                    _idf[q.Key] * (k1 + 1) * tf[q.Key] / (
                    k1 * (1.0 - b + b * _docLens[doc] / _avgDocLen) + tf[q.Key]) * (
                    k3 + 1) * q.Value / (k3 + q.Value)));
            i++;
        }


        return results;
    }

}
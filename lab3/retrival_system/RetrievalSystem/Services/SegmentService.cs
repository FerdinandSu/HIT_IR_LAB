using System.Text.Json.Serialization;
using System.Web;

namespace RetrievalSystem.Services;

/// <summary>
/// 基于Web的分词服务
/// </summary>
public class SegmentService
{
    private readonly HttpClient _client;
    private HashSet<string> StopWords { get; }

    public SegmentService(HttpClient client, IConfiguration config)
    {
        _client = client;
        StopWords = new HashSet<string>(File.ReadAllLines(config["stopWords"]));
    }

    private class ProcessResult
    {
        [JsonPropertyName("t")]
        public string T{ get; set; }
        [JsonPropertyName("p")]
        public double P { get; set; }
    }
    public async Task<string[]> ProcessAsync(string question)
    {

        var results=await _client.GetFromJsonAsync<ProcessResult[]>(
            $"http://114.67.84.223/get.php?source={HttpUtility.UrlEncode(question)}&param1=0&param2=1&json=1");
        return results!.Select(x=>x.T)
            .Where(t=>!StopWords.Contains(t))
            .ToArray();
    }
}
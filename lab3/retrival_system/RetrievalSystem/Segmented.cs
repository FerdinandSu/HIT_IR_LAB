using System.Text.Json.Serialization;

namespace RetrivalSystem;

public class Segmented
{
    public static Segmented Empty { get; } = new();
    [JsonPropertyName("url")] public string Url { get; init; } = String.Empty;
    [JsonPropertyName("title")] public string OriginTitle { get; init; } = String.Empty;
    [JsonPropertyName("segmented_title")] public string[] Title { get; init; } = Array.Empty<string>();

    [JsonPropertyName("segmented_file_name")]
    public Dictionary<string, string[]> Files { get; set; } = new();
    [JsonPropertyName("segmented_parapraghs")]
    public string[] Paragraphs { get; set; }
    [JsonPropertyName("paragraghs")]
    public string OriginParagraphs { get; set; }

    public DateOnly Date
    {
        get
        {
            var urlFragments = Url.Split('/');
            try
            {
                var urlDate =
                    new DateOnly(int.Parse(urlFragments[^4]),
                        int.Parse(urlFragments[^3]),
                        int.Parse(urlFragments[^2]));
                return urlDate;
            }
            catch (Exception)
            {
                return DateOnly.MinValue;
            }


        }
    }
}
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace CrawSharp;

public class Result
{
    public static Result Empty { get; }= new();
    [JsonPropertyName("url")]
    public string? Url{ get; init; }
    [JsonPropertyName("title")]
    public string? Title{ get; init; }
    [JsonPropertyName("file_name")]
    public List<string>? FileName { get; set; }
    [JsonPropertyName("paragraghs")]
    public string? Paragraphs{ get; set; }
}
using RetrivalSystem;

namespace RetrievalSystem.Models;

public class SearchResultModel
{
    public string Role { get; set; } = IdentityInfo.Guest;
    public List<Segmented>? CommonSearchResult { get; set; }
    public List<(Segmented, string)>? FileSearchResult { get; set; }
}
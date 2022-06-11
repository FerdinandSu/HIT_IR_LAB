using RetrivalSystem;

namespace RetrievalSystem.Services;

public class SearchService
{
    private readonly ILogger<SearchService> _logger;
    private readonly Dictionary<string, Segmented> _origin;
    private readonly SegmentService _segment;
    private readonly Bm25 _fileModel;
    private readonly Bm25 _commonModel;
    private readonly int _select;

    public SearchService(
        ILogger<SearchService> logger,
        Dictionary<string, Segmented> origin, IConfiguration config, SegmentService segment)
    {
        _logger = logger;
        _origin = origin;
        _segment = segment;
        _select = config.GetSection("Select").Get<int>();
        _logger.LogInformation("Getting Bm25:Common...");
        _commonModel = Bm25.Up(config["Bm25:Common"],
            () =>
                origin.Values.ToDictionary(o => o.OriginTitle, o => o.Title.Concat(o.Paragraphs).ToArray())
                    );
        _logger.LogInformation("Bm25:Common is Ready.");
        _logger.LogInformation("Getting Bm25:File...");
        _fileModel = Bm25.Up(config["Bm25:File"],
            () =>
                origin.Values.SelectMany(o =>
                        o.Files.Select(f => (o.OriginTitle + "@" + f.Key, o.Title.Concat(f.Value))))
                    .ToDictionary(o => o.Item1,
                        o => o.Item2.ToArray()));
        _logger.LogInformation("Bm25:File is Ready.");
        var now = DateTime.Now;
        var (year,month,date)=(now.Year,now.Month,now.Day);
        Predicts = new()
        {
            {IdentityInfo.Administrator, _ => true}, // 管理员拥有完整权限
            {
                IdentityInfo.VerifiedUser, GetPredictByDate(
                    new(year - 3, month, date))
            }, // 认证用户可查看3年
            {
                IdentityInfo.SignedUser,
                GetPredictByDate(
                    new(year - 1, month, date))
            }, // 登录用户可查看1年
            {
                IdentityInfo.Guest,
                GetPredictByDate(
                    new(year, month-1, date))
            }, // 来宾可查看1个月
        };
    }

    public async Task<IEnumerable<Segmented>> Search(string query, Func<Segmented, bool> predict)
    {
        var processed = await _segment.ProcessAsync(query);
        return _commonModel.ScoreOf(processed)
            .Select(sn => (_origin[sn.Item1], sn.Item2))
            .Where(v => predict(v.Item1))
            .OrderByDescending(tp => tp.Item2)
            .Take(_select)
            .Select(tp => tp.Item1);
    }
    public async Task<IEnumerable<(Segmented, string)>> SearchFile(string query, Func<Segmented, bool> predict)
    {
        var processed = await _segment.ProcessAsync(query);
        return _fileModel.ScoreOf(processed)
            .Select(k => (k.Item1.Split('@', 2), k.Item2))
            .Select(sn => (_origin[sn.Item1[0]], sn.Item1[1], sn.Item2))
            .Where(v => predict(v.Item1))
            .OrderByDescending(tp => tp.Item3)
            .Take(_select)
            .Select(tp => (tp.Item1, tp.Item2));
    }

    public Func<Segmented, bool> GetPredictByDate(DateOnly date)
    {
        //URL格式:http://today.hit.edu.cn/article/{yyyy}/{MM}/{DD}/*****
        return seg => seg.Date >= date;

    }

    public Dictionary<string, Func<Segmented, bool>> Predicts{ get; }
}
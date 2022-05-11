using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.Encodings.Web;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;
using System.Text.Unicode;
using System.Threading.Channels;
using System.Web;
using HitRefresh.HitGeneralServices.CasLogin;
using HtmlAgilityPack;

namespace CrawSharp;

public class Executor
{
    static readonly string[] AppendixExtensions = {
        ".xls",
        ".doc",
        ".docx",
        ".txt",
        ".xlsx",
        ".pdf"
    };

    private const int ThreadCount = 32;
    private const int Target = 1000;
    private const int TargetAppendix = 100;
    private readonly Channel<string> _queue = Channel.CreateUnbounded<string>();
    private int _succeeded = 0;
    private int _hasAppendix = 0;
    private readonly HashSet<string> _legalUrls = new() { "today.hit.edu.cn" };
    readonly ConcurrentDictionary<string, Result> _visit = new();
    readonly LoginHttpClient _client = new();

    public async Task Login(string username, string password)
    => await _client.LoginAsync(username, password);
    public async Task StartWith(string startUrl)
    {
        var html = new HtmlDocument();
        try
        {
            html.Load(await _client.GetStreamAsync(startUrl));
        }
        catch (Exception)
        {
            return;
        }
        // 信息处理：获取标题、正文、附件

        var urls =
            html.DocumentNode.SelectNodes("//a")
                .Select(n => n.GetAttributeValue<string>("href", ""))
                .Where(link => link.StartsWith("http://")
                               && _legalUrls.Contains(link.Split('/')[2])
                               && !_visit.ContainsKey(link))
                .ToHashSet();
        foreach (var next in
                 urls.Where(
                     next => next.StartsWith("http://today.hit.edu.cn/article/")
                             && !_visit.ContainsKey(next)))
        {
            await _queue.Writer.WriteAsync(next);
        }
    }
    public async Task Execute()
    {
        ThreadPool.SetMinThreads(ThreadCount, ThreadCount);
        await Task.WhenAll(
            Enumerable.Range(0, ThreadCount)
                .Select(_ => DoExecute())
                .Append(Monitor()));
    }

    private async Task Monitor()
    {
        //await Task.Yield();
        var sw = new Stopwatch();
        sw.Start();
        for (; _hasAppendix < TargetAppendix ||
               _succeeded < Target;)
        {
            Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] {sw.Elapsed} Passed, {_hasAppendix}/{TargetAppendix} & {_succeeded}/{Target} Done.");
            await Task.Delay(1000);

        }
    }
    private async Task DoExecute()
    {
        await Task.Yield();
        for (;
             _hasAppendix < TargetAppendix ||
             _succeeded < Target;)
        {

            if (!_queue.Reader.TryRead(out var top))
            {
                await Task.Delay(100);
                continue;
            }
            await CrawUrl(top);
        }


    }
    private async Task CrawUrl(string url)
    {
        if (!_visit.TryAdd(url,Result.Empty)) return;
        var html = new HtmlDocument();
        try
        {
            var r = await _client.GetAsync(url);
            if (r.StatusCode == System.Net.HttpStatusCode.Found)
            {
                r = await _client.GetAsync(r.Headers.Location);
            }

            if (!r.IsSuccessStatusCode)
            {
                _visit.TryRemove(url, out _);
                return;
            }
                
            html.Load(await r.Content.ReadAsStreamAsync());
        }
        catch (Exception)
        {
            _visit.TryRemove(url, out _);
            return;
        }
        // 信息处理：获取标题、正文、附件
        var titleTag = html.DocumentNode.SelectSingleNode(
            //"//h1"
            "//h3"//今日哈工大用h3作文章标题，xs
            );
        if (titleTag != null &&
            !string.IsNullOrWhiteSpace(titleTag.InnerText))
        {
            var title = titleTag.InnerText.Trim();
            var paragraphs =
                string.Join(' ',
                html.DocumentNode.SelectNodes("//p")
                    .Select(n =>
                        n.InnerText.Replace("\n", "")
                            .Replace(" ", "")));
            var urls =
                html.DocumentNode.SelectNodes("//a")
                    .Select(n => n.GetAttributeValue<string>("href", ""))
                    .Where(link => link.StartsWith("http://")
                                   && _legalUrls.Contains(link.Split('/')[2])
                                   && !_visit.ContainsKey(link))
                    .ToHashSet();
            var appendixUrls =
                urls.Where(u =>
                    AppendixExtensions.Any(u.EndsWith))
                    .ToList();
            var appendix = new List<string>();
            foreach (var appendixUrl in appendixUrls)
            {
                urls.Remove(appendixUrl);
                try
                {
                    var resp =
                        await _client.GetAsync(appendixUrl, HttpCompletionOption.ResponseHeadersRead);
                    if (!resp.IsSuccessStatusCode)
                    {
                        _visit.TryRemove(url, out _);
                        return;
                    }
                    var fileName = appendixUrl.Split('/')[^1];
                    var fn = HttpUtility.UrlDecode(fileName);
                    appendix.Add(fn);
                    await using var fs = File.Create(Path.Combine("files", fn));
                    var webFs = await resp.Content.ReadAsStreamAsync();
                    await webFs.CopyToAsync(fs);
                }
                catch (Exception)
                {
                    return;
                }

            }

            if (
                _visit.TryUpdate(url,new()
                {
                    Url = url,
                    FileName = appendix,
                    Paragraphs = paragraphs,
                    Title = title
                },Result.Empty))
            {
                Interlocked.Increment(ref _succeeded);
                if (appendix.Count > 0) Interlocked.Increment(ref _hasAppendix);
                foreach (var next in
                         urls.Where(
                             next => next.StartsWith("http://today.hit.edu.cn/article/")
                                            && !_visit.ContainsKey(next)))
                {
                    await _queue.Writer.WriteAsync(next);
                }
            }


        }

    }

    public async Task Export(string path)
    {
        await JsonSerializer.SerializeAsync<IEnumerable<Result>>(
            File.Open(path, FileMode.CreateNew, FileAccess.Write),
            _visit.Values, new JsonSerializerOptions
            {
                Encoder = JavaScriptEncoder.Create(UnicodeRanges.All)
            });
    }
}
using Microsoft.AspNetCore.Mvc;
using RetrievalSystem.Models;
using System.Diagnostics;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.StaticFiles;
using RetrievalSystem.Services;
using static System.IO.File;

namespace RetrievalSystem.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;
        private readonly UserManager<IdentityUser> _userManager;
        private readonly SearchService _search;
        private readonly IContentTypeProvider _typeProvider;
        private readonly string _fileDir;

        public HomeController(ILogger<HomeController> logger,IConfiguration config,
            UserManager<IdentityUser> userManager, SearchService search,
            IContentTypeProvider typeProvider)
        {
            _logger = logger;
            _userManager = userManager;
            _search = search;
            _typeProvider = typeProvider;
            _fileDir = config["files"];
        }

        public IActionResult File(string fileName)
        {
            var path=Path.Combine(_fileDir,fileName);
            if(!Exists(path))return NotFound();
            return File( OpenRead(path),
                _typeProvider.TryGetContentType(
                    fileName, out var mime) ? mime : "",
                fileName, true);
        }
        public IActionResult Index()
        {
            return View();
        }

        private async Task< string> GetRoleName()
        {

            var user = await _userManager.GetUserAsync(User);
            if (user is null) return IdentityInfo.Guest;
            var roles = await _userManager.GetRolesAsync(user);
            return roles.FirstOrDefault(IdentityInfo.SignedUser);
        }
        [HttpPost]
        public async Task<IActionResult> Search(string? searchWord)
        {
            var role = await GetRoleName();
            return View("Index",new SearchResultModel
            {
                CommonSearchResult = (await _search.Search(searchWord ?? "",
                    _search.Predicts[role]
                )).ToList(),
                Role = role

            });
        }
        [HttpPost]
        public async Task<IActionResult> SearchFile(string? searchWord)
        {
            var role = await GetRoleName();
            return View("Index", new SearchResultModel
            {
                FileSearchResult = (await _search.SearchFile(searchWord ?? "",
                    _search.Predicts[role]
                )).ToList(),
                Role=role
            });
        }
        public IActionResult Privacy()
        {
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}
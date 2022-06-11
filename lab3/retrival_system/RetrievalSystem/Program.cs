using System.Text.Json;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.StaticFiles;
using Microsoft.EntityFrameworkCore;
using RetrievalSystem;
using RetrievalSystem.Data;
using RetrievalSystem.Services;
using RetrivalSystem;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
var connectionString = builder.Configuration.GetConnectionString("DefaultConnection");
builder.Services.AddDbContext<ApplicationDbContext>(options =>
    options.UseSqlServer(connectionString));
builder.Services.AddDatabaseDeveloperPageExceptionFilter();

builder.Services.AddDefaultIdentity<IdentityUser>(options => {
        options.SignIn.RequireConfirmedAccount = false;
        options.Password.RequireDigit = false;
        options.Password.RequireNonAlphanumeric = false;
        options.Password.RequireUppercase = false;
        options.Password.RequireLowercase = false;
        options.Password.RequiredLength = 1;
    })
    .AddRoles<IdentityRole>()
    .AddEntityFrameworkStores<ApplicationDbContext>();
builder.Services.AddControllersWithViews();
builder.Services.AddSingleton(isp =>
{
    var r=new Dictionary<string, Segmented>();
    var origin= JsonSerializer.Deserialize<Segmented[]>(File.ReadAllText(
            isp.GetRequiredService<IConfiguration>()["origin"]))!;
    foreach (var segmented in origin)
    {
        r[segmented.OriginTitle] = segmented;
    }
    return r;
});
builder.Services
    .AddSingleton<SearchService>()
    .AddSingleton<SegmentService>()
    .AddHttpClient();
builder.Services.AddSingleton<IContentTypeProvider, FileExtensionContentTypeProvider>();
var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseMigrationsEndPoint();
}
else
{
    app.UseExceptionHandler("/Home/Error");
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();

app.UseRouting();

app.UseAuthentication();
app.UseAuthorization();

app.MapControllerRoute(
    name: "default",
    pattern: "{controller=Home}/{action=Index}/{id?}");
app.MapRazorPages();
var scope=app.Services.CreateScope();

var db = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();
await db.Database.MigrateAsync();

var roleManager = scope.ServiceProvider.GetRequiredService<RoleManager<IdentityRole>>();
var admin = await roleManager.FindByNameAsync(IdentityInfo.Administrator);
if (admin == null)
{
    var userManager = scope.ServiceProvider.GetRequiredService<UserManager<IdentityUser>>();
    foreach (var role in IdentityInfo.Roles)
    {
        await IdentityInfo.CreateDefault(role,userManager,roleManager);
    }

}
scope.Dispose();

app.Run();

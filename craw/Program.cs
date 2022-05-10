using CrawSharp;

var x = new Executor();

string? username, password;
if (args.Length > 0)
{
   (username,password)=(args[0],args[1]);
}
else
{
    Console.WriteLine("输入学号");
    username = Console.ReadLine();
    Console.WriteLine("输入密码");
    password = Console.ReadLine();
}

if (username == null || password == null) return;

await x.Login(username, password);
Console.WriteLine("Login Completed.");
Directory.CreateDirectory("files");
await x.StartWith("http://today.hit.edu.cn/");
Console.WriteLine("Url Set Ready.");
await x.Execute();
Console.WriteLine("Craw Done.");
await x.Export("craw.json");
Console.WriteLine("Done.");

using Microsoft.AspNetCore.Identity;
using RetrievalSystem.Data;

namespace RetrievalSystem;

public class IdentityInfo
{
    public static readonly string[] Roles =
    {
        Administrator,
        VerifiedUser,
        SignedUser,
        Guest
    };
    public const string Administrator = nameof(Administrator);
    public const string VerifiedUser = nameof(VerifiedUser);
    public const string SignedUser = nameof(SignedUser);
    public const string Guest = nameof(Guest);

    public static async Task CreateDefault(string roleName, UserManager<IdentityUser> um,
        RoleManager<IdentityRole> rm)
    {
        try
        {
            var roleDefault = new IdentityUser($"{roleName}")
            {
                Email = $"{roleName[0]}@em.co"
            };
            await rm.CreateAsync(new(roleName));
            var r= await um.CreateAsync(roleDefault, "HiT%123");
            await um.AddToRoleAsync(roleDefault, roleName);
        }
        catch (Exception e)
        {
            Console.WriteLine(e);
            throw;
        }

    }
}
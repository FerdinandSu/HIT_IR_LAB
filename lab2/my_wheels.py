def count_if(seq, pred):
    count = 0
    for e in seq:
        if pred(e):
            count += 1
    return count


def max_with(seq, key_selector, min_value=0):
    e = None
    for i in seq:
        current = key_selector(i)
        if(current > min_value):
            min_value = current
            e = i
    return e


def size_of_longest_common_sequence(s1, s2):
    s1_size = len(s1)
    s2_size = len(s2)
    s1 = " "+s1
    s2 = " "+s2
    dp = [[0] * (s2_size + 1)] * (s1_size + 1)
    for i in range(1, s1_size+1):
        for j in range(1, s2_size+1):
            if s1[i] == s2[j]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[s1_size][s2_size]


def size_of_longest_common_substring(s1, s2):
    s1_size = len(s1)
    s2_size = len(s2)
    s1 = " "+s1
    s2 = " "+s2
    dp = [[0] * (s2_size + 1)] * (s1_size + 1)
    max_length = 0
    for i in range(1, s1_size+1):
        for j in range(1, s2_size+1):
            if s1[i] == s2[j]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
            else:
                dp[i][j] = 0
    return max_length

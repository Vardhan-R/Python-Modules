def convertNum(num: str, from_base: int, to_base: int, min_op_str_len: int = 1): # min_op_str_len ==> minimum output string length
    l = []
    f = sum([(ord(num[x]) - 48 - 7 * int(ord(num[x]) in range(65, 91))) * from_base ** (len(num) - x - 1) for x in range(len(num))]) # [48, 57] ==> [0, 9]; [65, 90] ==> [10, 35]
    while f != 0:
        l.append(f % to_base)
        f //= to_base
    return "".join(["0" for x in range(min_op_str_len - len(l))]) + "".join([chr(l[x] + 48 + 7 * int(l[x] > 9)) for x in range(len(l) - 1, -1, -1)]) # [0, 9] ==> [48, 57]; [10, 35] ==> [65, 90]

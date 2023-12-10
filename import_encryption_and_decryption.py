import random
import import_number_system_converter as nsc

def xor(a, b):
    return int(a != b)

def xor_lst(lst, sk):
    xor_str = ""
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            xor_str += str(xor(lst[i][j], sk[j]))
    return xor_str

def encrypt(input_string, secret_key = None):
    bin_ascii_str = ""
    for i in input_string:
        bin_ascii_str += nsc.convertNum(ord(i), 10, 2, 8)

    bin_ascii_lst = []
    while len(bin_ascii_str) > 256:
        bin_ascii_lst.append(bin_ascii_str[:256])
        bin_ascii_str = bin_ascii_str[256:]
    if len(bin_ascii_str):
        bin_ascii_lst.append(bin_ascii_str)

    if secret_key:
        secret_key = str(secret_key)

        encrypted_bin_str = xor_lst(bin_ascii_lst, secret_key)
        return nsc.convertNum(encrypted_bin_str, 2, 36)
    else:
        secret_key = ""
        for i in range(256):
            secret_key += random.choice(["0", "1"])

        encrypted_bin_str = xor_lst(bin_ascii_lst, secret_key)

        return [nsc.convertNum(encrypted_bin_str, 2, 36), secret_key]

def decrypt(input_string, secret_key):
    encrypted_bin_str = nsc.convertNum(input_string, 36, 2)
    while len(encrypted_bin_str) % 8:
        encrypted_bin_str = "0" + encrypted_bin_str

    encrypted_bin_lst = []
    while len(encrypted_bin_str) > 256:
        encrypted_bin_lst.append(encrypted_bin_str[:256])
        encrypted_bin_str = encrypted_bin_str[256:]
    if len(encrypted_bin_str):
        encrypted_bin_lst.append(encrypted_bin_str)

    bin_ascii_str = xor_lst(encrypted_bin_lst, secret_key)

    decrypted_str = ""
    for i in range(int(len(bin_ascii_str) / 8)):
        decrypted_str += chr(int(nsc.convertNum(bin_ascii_str[8 * i: 8 * (i + 1)], 2, 10)))

    return decrypted_str
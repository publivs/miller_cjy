import base64
from Cryptodome.Cipher import AES

# 需要补位，str不是16的倍数那就补足为16的倍数
def add_to_16(value):
    while len(value) % 16 != 0:
        value += '\0'
    return str.encode(value)


# 加密方法
def aes_encrypt(key, t, iv):
    aes = AES.new(add_to_16(key), AES.MODE_CBC, add_to_16(iv))  # 初始化加密器
    encrypt_aes = aes.encrypt(add_to_16(t))                    # 先进行 aes 加密
    encrypted_text = str(base64.encodebytes(encrypt_aes), encoding='utf-8')  # 执行加密并转码返回 bytes
    return encrypted_text


# 解密方法
def aes_decrypt(key, t, iv):
    aes = AES.new(add_to_16(key), AES.MODE_CBC, add_to_16(iv))         # 初始化加密器
    base64_decrypted = base64.decodebytes(t.encode(encoding='utf-8'))  # 优先逆向解密 base64 成 bytes
    decrypted_text = str(aes.decrypt(base64_decrypted), encoding='utf-8').replace('\0', '')  # 执行解密密并转码返回str
    return decrypted_text


if __name__ == '__main__':
    secret_key = '12345678'   # 密钥
    text = '%5B%22%5C%22484%5C%22%22%2C%22%5C%229490%5C%22%22%5D'  # 加密对象
    iv = secret_key           # 初始向量
    encrypted_str = aes_encrypt(secret_key, text, iv)
    print('加密字符串：', encrypted_str)
    decrypted_str = aes_decrypt(secret_key, encrypted_str, iv)
    print('解密字符串：', decrypted_str)


# 加密字符串：lAVKvkQh+GtdNpoKf4/mHA==
# 解密字符串：I love Python!
'''
XhjNxekyLKlag0uh40AR2QVNna+v0VJPqntAPUwHa4MmtZRzCYPbzysv8WAPTbrLjS7DplawC6/EIFV+lTaaBrtdStfmVXp8gyBN2cPj8euplRalrtf4d9HrFUBZrunGA8l2+6WWTJH7XcbEa50F2n55IQXwRttAWXlvGGZM3o4=
KVPQuCJc6Zr7LORkmCfiP59E9wH7c9mNWl5mJ/G9qI5iKyZwxJXwvS7a0JLgYSyaD8/+jSu/7lhzKp686s+l93/DeWCGCwJfEIJoEPvx7KixLXmXs5AEbkEZECEy5YzbRONGePtVam/wI3MS5x1rQ5crlp/UmKLfFscj5iTubeI=
nlmKE7ZNFTCtu6Syf1NQ5napuC1x0yNyr5/Fz8Yu28NheR88OnnmDePPNm4J62LtyWLjyUt089/mbubIMogABXXhlUXbDfLDe4/tre15/KFFkz0DbSYUgo0ZgDXazCDilbp4AC3adpepbYpLabs+TtA7oLNIbYjmtVFHddShDX8=
XyuMBQkA+CsDirr0qKwuKTXehc14aG6ucaOxYTaU+l4PXJ/h8EQApQF+JU8VuEI9HQyAAN0rey713+h13X/OjKKfYuc6xLBNn63+bIvDCq7zhTVTVulZASMNmPvhnxgQncxKp8XhXlJcDIDCnFLd8ayVWUe9paffWJFDvz1j1LA=
'''
import base64
from Cryptodome.Cipher import ARC4


def rc4_encrypt(key, t):
    enc = ARC4.new(key.encode('utf8'))
    res = enc.encrypt(t.encode('utf-8'))
    res = base64.b64encode(res)
    return res


def rc4_decrypt(key, t):
    data = base64.b64decode(t)
    enc = ARC4.new(key.encode('utf8'))
    res = enc.decrypt(data)
    return res


if __name__ == "__main__":
    secret_key = '12345678'   # 密钥
    text = '%5B%22%5C%22484%5C%22%22%2C%22%5C%229490%5C%22%22%5D'  # 加密对象
    encrypted_str = rc4_encrypt(secret_key, text)
    print('加密字符串：', encrypted_str)
    '''
    bLIfakBR7PQJ34R1zmYPLwjCYYowhWDZayNi3DvDrNLebmsVeI/F6mP5o+GJsHu6tHf9ZJ2tHQY/ZVPFHnSNHOJR56Fql7/qkK6Rrj8izdLdik/LTn3pmp/9BYYNiyue1SwLgNpQlyiRUXJPyd2QZM955oI3376bK43T4MROVlk=
    '''
    decrypted_str = rc4_decrypt(secret_key, encrypted_str)
    print('解密字符串：', decrypted_str)
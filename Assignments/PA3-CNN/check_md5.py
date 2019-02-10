import hashlib
import os

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

check = [
"fe8ed0a6961412fddcbb3603c11b3698",
"ab07a2d7cbe6f65ddd97b4ed7bde10bf",
"2301d03bde4c246388bad3876965d574",
"9f1b7f5aae01b13f4bc8e2c44a4b8ef6",
"1861f3cd0ef7734df8104f2b0309023b",
"456b53a8b351afd92a35bc41444c58c8",
"1075121ea20a137b87f290d6a4a5965e",
"b61f34cec3aa69f295fbb593cbd9d443",
"442a3caa61ae9b64e61c561294d1e183",
"09ec81c4c31e32858ad8cf965c494b74",
"499aefc67207a5a97692424cf5dbeed5",
"dc9fda1757c2de0032b63347a7d2895c"
]

for root, dirs, files in os.walk("./data/", topdown=False):
    for i, file in enumerate(files):
        print(check[i] == md5(root + file))
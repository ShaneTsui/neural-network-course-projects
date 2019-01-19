def f(x, d):
    return 1 - (1 - x) ** d

xs = [0.01, 0.02]
ds = [2, 10, 1000]

for x in xs:
    for d in ds:
        print("{} & {} & {} \\\\".format(x, d, f(x, d)))
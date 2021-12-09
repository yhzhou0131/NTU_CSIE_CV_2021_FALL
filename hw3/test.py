n = 8

for i in range(n):
    print(" " * (n-1-i), end="")
    for j in range(i, 0, -1):
        print(j, end="")
    for j in range(0, i + 1):
        print(j, end="")
    print()
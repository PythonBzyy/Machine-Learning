n = int(input())
grid = [list(map(int, input().strip())) for _ in range(n)]

# 初始化记录每一列清扫状态的列表
cleaned = [False] * n

# 遍历每一行，选择清扫可以使得更多行变得干净的列
for i in range(n):
    max_cleaned = -1
    clean_col = -1
    for j in range(n):
        if not cleaned[j]:
            cleaned_rows = 0
            for k in range(n):
                if grid[k][j] == grid[k][0] == grid[i][0]:
                    cleaned_rows += 1
            if cleaned_rows > max_cleaned:
                max_cleaned = cleaned_rows
                clean_col = j
    cleaned[clean_col] = True

# 计算清扫后干净的行数
clean_rows = 0
for i in range(n):
    is_clean = True
    for j in range(n):
        if grid[i][j] != grid[i][0]:
            is_clean = False
            break
    if is_clean:
        clean_rows += 1

print(clean_rows)

data = [5, 5, 5, 5, 3, 3, 4, 4, 4, 7, 7, 7]

# 创建一个字典来记录每个数字的索引
index_dict = {}
index = 0
for num in data:
    if num not in index_dict:
        index_dict[num] = index
        index += 1

# 使用字典将原始数据映射为新的值
new_data = [index_dict[num] for num in data]
print(new_data)
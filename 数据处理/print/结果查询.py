input_file_path = "记录.txt"   # 原始文件路径
output_file_path = "filtered_info.txt"  # 结果保存路径

# 打开原始文件并读取符合条件的行
with open(input_file_path, 'r', encoding='utf-8') as infile:
    filtered_lines = [line for line in infile if line.startswith("INFO:root:\tIter:")]

# 将提取的行写入新文件
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    outfile.writelines(filtered_lines)

print(f"已提取 {len(filtered_lines)} 条记录，并保存到 {output_file_path}")

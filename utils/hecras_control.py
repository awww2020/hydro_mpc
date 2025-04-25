import numpy as np
import win32com.client as win32
import re
import os

def modify_unsteady_file(sim_id, q_upstream, q_downstream, q_0, z_0, x0):

    original_u_filename = f"E:\program\hec_ras_project\mpc_test_1\mpc_test.u{sim_id:02d}"
    with open(original_u_filename, "r") as f:
        lines = f.readlines()

    # 1. 确保flow title 正确
    for i, line in enumerate(lines):
        if line.startswith("Flow Title="):
            expected = f"Flow Title=unsteady flow {sim_id:02d}\n"
            if line != expected:
                raise ValueError(
                    f"Unexpected Flow Title: {line.strip()!r}, "
                    f"expected {expected.strip()!r}"
                )
            break

    # 修改 Initial Flow Loc 和 Initial RRR Elev 行
    '''
    # 只关注这四个 elevation
    ele_targets = {"4000", "3000.00*", "2000.00*", "1000.00*"}

    new_lines = []
    for line in lines:
        # 跳过所有 Initial Flow Loc 行
        if line.startswith("Initial Flow Loc=channel"):
            continue

        # 处理 Initial RRR Elev 行
        if line.startswith("Initial RRR Elev=channel"):
            parts = line.split(',')
            # 检查第三列是否在我们指定的四个 elevation 里
            if len(parts) >= 4 and parts[2].strip() in ele_targets:
                try:
                    wl = next(water_level_iterator)
                except StopIteration:
                    raise RuntimeError("water_level_iterator 中的值不足 4 个")
                # 用新的水位替换第四列，并保留换行符
                parts[3] = f"{wl:10.6f}\n"
                new_lines.append(','.join(parts))
            # 如果不是那四个 elevation，就跳过（即删除）
            continue
            
        # 其它行原样保留
        new_lines.append(line)
            # 最终把 lines 指向新的列表
    lines = new_lines
    '''
    q_0_iterator = iter(q_0)
    z_0_iterator = iter(z_0)

    for i, line in enumerate(lines):
        if line.startswith("Initial Flow Loc=channel"):
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    # 取出下一个流量值
                    q = next(q_0_iterator)
                except StopIteration:
                    print("No more q_0 values")
                    break
                # 格式化到 6 位小数，并加上换行
                parts[3] = f"{q:10.6f}\n"
                lines[i] = ','.join(parts)

        elif line.startswith("Initial RRR Elev=channel"):
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    z = next(z_0_iterator)
                except StopIteration:
                    print("No more z_0 values")
                    break
                parts[3] = f"{z:10.6f}\n"
                lines[i] = ','.join(parts)


    # 2. 更新Flow Hydrograph的个数和数据行
    found = 0
    for i in range(len(lines)):
        if lines[i].startswith("Flow Hydrograph="):
            if found == 0:
                flow_count = len(q_upstream)
                lines[i] = f"Flow Hydrograph= {flow_count}\n"  # 更新计数 [^1]
                lines[i + 1:i + 2] = [  # 替换为多行
                    " ".join(f"{q_up:7.3f}" for q_up in q_upstream[j:j + 5]) + "\n"
                    for j in range(0, flow_count, 10)
                ]

                found += 1
            elif found == 1:
                stage_count = len(q_downstream)
                lines[i] = f"Flow Hydrograph= {stage_count}\n"  # 更新计数 [^3]
                lines[i + 1:i + 2] = [  # 替换为多行
                    " ".join(f"{q_down:7.3f}" for q_down in q_downstream[j:j + 5]) + "\n"
                    for j in range(0, stage_count, 10)
                ]
                found += 1

    for i in range(len(lines)):
        if lines[i].startswith("Flow Hydrograph Inital WS="):
            lines[i] = f"Flow Hydrograph Inital WS={x0[4:5].item()}\n"

    # 保存新文件
    new_u_filename = f"E:\program\hec_ras_project\mpc_test_1\mpc_test.u{sim_id:02d}"
    with open(new_u_filename, "w", encoding='utf-8') as f:
        f.writelines(lines)
    # print('成功生成新.u文件，new_u_filename', new_u_filename)
    new_u_filename = f"E:\program\hec_ras_project\mpc_test_1\ceshi.txt"
    with open(new_u_filename, "w", encoding='utf-8') as f:
        f.writelines(lines)
    # print('成功生成新.u文件，new_u_filename', new_u_filename)

    # 读取 HEC-RAS 计划（.prj）文件， 检查现有行中是否包含该Unsteady File
    file_path = r'E:\program\hec_ras_project\mpc_test_1\mpc_test.prj'
    with open(file_path, 'r', encoding='utf-8') as f:
        original_lines = f.read().splitlines()  # 逐行读取

    new_Unsteady_entry = f"Unsteady File=u{sim_id:02d}"  # 当前要添加的Unsteady File行内容
    exists = False  # 标记是否已存在当前sim_id的Unsteady File
    last_Unsteady_pos = -1  # 记录最后一个Unsteady File行的位置

    for idx, line in enumerate(original_lines):
        stripped_line = line.strip()
        if stripped_line == new_Unsteady_entry:
            exists = True  # 发现完全匹配的Unsteady File行
        if stripped_line.startswith("Unsteady File="):
            last_Unsteady_pos = idx  # 记录最后一个Unsteady File的位置

    # 如果不存在，则在恰当位置插入
    if not exists:
        new_lines = []
        for idx, line in enumerate(original_lines):
            # 追加当前行到新列表
            new_lines.append(line)
            # 如果当前行是最后一个Plan File行，则在其后插入新条目
            if idx == last_Unsteady_pos:
                new_lines.append(new_Unsteady_entry)
                # print('在prj新增Unsteady File:', new_Unsteady_entry)
        # 覆盖原文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
    else:
        pass
        # print('Unsteady File已存在，无需重复添加')

    return new_u_filename

def create_plan_file(sim_id, template_plan_path):
    plan_title = f"plan {sim_id:02d}"
    # 读取模板内容
    with open(template_plan_path, 'r') as f:
        plan_content = f.read()
    # 替换Plan Title和Flow File行
    plan_content = re.sub(r'^Plan Title=.*$', f'Plan Title={plan_title}', plan_content, flags=re.MULTILINE)
    plan_content = plan_content.replace('Flow File=u02', f'Flow File=u{sim_id:02d}')
    plan_content = re.sub(r'^Short Identifier=.*$', f'Short Identifier={sim_id:02d}', plan_content, flags=re.MULTILINE)
    plan_content = re.sub(r'^Simulation Date=.*$', 'Simulation Date=22SEP2008,00:00:00,22SEP2008,00:04:00',
                          plan_content, flags=re.MULTILINE)
    # 根据需要替换其他参数，比如Geom File等
    new_p_filename = os.path.abspath(f"E:/program/hec_ras_project/mpc_test_1/mpc_test.p{sim_id:02d}")
    # print('成功生成新.p文件，new_u_filename', new_p_filename)
    with open(new_p_filename, 'w') as f:
        f.write(plan_content)

    # 读取 HEC-RAS 计划（.prj）文件
    file_path = r'E:\program\hec_ras_project\mpc_test_1\mpc_test.prj'
    with open(file_path, 'r', encoding='utf-8') as f:
        original_lines = f.read().splitlines()  # 逐行读取

    new_plan_entry = f"Plan File=p{sim_id:02d}"  # 当前要添加的Plan File行内容
    exists = False  # 标记是否已存在当前sim_id的Plan File
    last_plan_pos = -1  # 记录最后一个Plan File行的位置

    # 检查现有行中是否包含该Plan File，并记录最后一个Plan File的位置
    for idx, line in enumerate(original_lines):
        stripped_line = line.strip()
        if stripped_line == new_plan_entry:
            exists = True  # 发现完全匹配的Plan File行
        if stripped_line.startswith("Plan File="):
            last_plan_pos = idx  # 记录最后一个Plan File的位置

    # 如果不存在，则在恰当位置插入
    if not exists:
        new_lines = []
        for idx, line in enumerate(original_lines):
            # 追加当前行到新列表
            new_lines.append(line)
            # 如果当前行是最后一个Plan File行，则在其后插入新条目
            if idx == last_plan_pos:
                new_lines.append(new_plan_entry)
                # print('在prj新增Plan File:', new_plan_entry)
        # 覆盖原文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
    else:
        pass
        # print('Plan File已存在，无需重复添加')

    return new_p_filename, plan_title
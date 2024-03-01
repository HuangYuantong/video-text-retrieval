import os


def delete_repeat_lines(file_path: str):
    if os.path.isdir(file_path):
        files = os.listdir(file_path)
        for file in files:
            if 'log' in file and '.txt' in file:
                file_path = os.path.join(file_path, file)
                break

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read()

    lines = lines.replace('\n\n======================================\n', '').split('\n')

    new_lines = list()
    for idx, line in enumerate(lines):
        if idx + 1 >= len(lines) or line != lines[idx + 1]:
            new_lines.append(line)

    print('\n'.join(new_lines))
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))


if __name__ == '__main__':
    delete_repeat_lines(input('请输入要修改的log文件（或文件夹）：'))

import ast
import json
import os


def convert_md_to_json(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    try:
        data = ast.literal_eval(content)
    except Exception as e:
        print(f'转换失败！请检查.md文件内容是否为python字典格式。错误：{e}')
        return

    json_str = json.dumps(data, ensure_ascii=False, indent=2)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json_str)

    print(f'转换完成！结果保存在: {output_path}')


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = r"C:\Users\xuchi\Desktop\SwiftMart\outputs.md"
    output_file = r"C:\Users\xuchi\Desktop\SwiftMart\outputs.json"

    convert_md_to_json(input_file, output_file)

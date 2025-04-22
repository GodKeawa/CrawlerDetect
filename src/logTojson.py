import re
import json
import ijson
from datetime import datetime
import urllib.parse


def parse_log_line(line):
    # 使用正则表达式解析日志行的各个部分
    pattern = r'(\S+) (\S+) (\S+) \[(.*?)\] "(.*?)" (\d+) (\d+) "(.*?)" "(.*?)" "(.*?)"'
    match = re.match(pattern, line)

    if not match:
        print("fail to match!")
        return None

    ip, _, user, time_str, request, status, size, referer, user_agent, _ = match.groups()

    # 解析请求部分
    request_parts = request.split(' ')
    if len(request_parts) == 3:
        method, path, protocol = request_parts
    elif len(request_parts) == 2:
        method, path = request_parts
        protocol = ""
    elif len(request_parts) == 1:
        method = request_parts[0] if request_parts else ""
        path = ""
        protocol = ""
    elif len(request_parts) == 0:
        method, path, protocol = "", "", ""
    else:
        method, path, protocol = request_parts[0], ' '.join(request_parts[1:-1:]), request_parts[-1]

    # 解析时间
    try:
        dt = datetime.strptime(time_str, "%d/%b/%Y:%H:%M:%S %z")
        time_formatted = dt.isoformat()
    except ValueError:
        time_formatted = time_str

    # URL解码路径
    try:
        decoded_path = urllib.parse.unquote(path)
    except:
        decoded_path = path

    # 构建JSON对象
    log_json = {
        "ip": ip,
        "user": user if user != "-" else None,
        "timestamp": time_formatted,
        "method": method,
        "path": path,
        "decoded_path": decoded_path,
        "protocol": protocol,
        "status": int(status),
        "size": int(size),
        "referer": referer if referer != "-" else None,
        "user_agent": user_agent if user_agent != "-" else None
    }

    return log_json

def parse_log_file(file_path):
    """解析整个日志文件并返回JSON对象列表"""
    logs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            log_json = parse_log_line(line.strip())
            if log_json:
                logs.append(log_json)
    return logs

def convert_log_format(input_file, output_file):
    """
    将新格式的日志数据转换为当前数据集格式

    参数:
        input_file: 输入文件路径（新格式JSON）
        output_file: 输出文件路径
    """
    converted_logs = []
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = ijson.kvitems(f, '')

        # 处理每条日志记录
        for key, log_entry in data:
            # 直接从日志条目中获取所需信息
            try:
                # 优先使用日志条目中的字段
                converted_log = {
                    "ip": log_entry.get("ip", ""),
                    "user": None,  # 原日志中无用户信息
                    "timestamp": log_entry.get("timestamp", "")[:-5:] + "+00:00",
                    "method": log_entry.get("method", ""),
                    "path": log_entry.get("resource", ""),
                    "decoded_path": log_entry.get("resource", ""),  # 无法解码哈希路径
                    "protocol": "HTTP/1.1",  # 默认值
                    "status": int(log_entry.get("response", "0")),
                    "size": int(log_entry.get("bytes", "0")),
                    "referer": log_entry.get("referrer", None) if log_entry.get("referrer", "-") != "-" else None,
                    "user_agent": log_entry.get("useragent", None)
                }

                converted_logs.append(converted_log)
            except Exception as e:
                print(f"处理日志时出错: {e}, 日志: {log_entry}")

    # 保存到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_logs, f, ensure_ascii=False, indent=2)

    print(f"转换完成，共处理 {len(converted_logs)} 条日志记录。")


def test():
    _parsed_logs = parse_log_file("../tests/example.txt")
    with open("../tests/parsed_logs.json", "w", encoding="utf-8") as f:
        json.dump(_parsed_logs, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # parsed_logs = parse_log_file("../data/train.log")
    # with open("../data/train.json", "w", encoding="utf-8") as f:
    #     json.dump(parsed_logs, f, ensure_ascii=False, indent=2)
    convert_log_format("../data/tests/public_v2.json", "../data/test.json")
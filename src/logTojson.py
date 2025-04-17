import re
import json
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

def test():
    _parsed_logs = parse_log_file("../tests/example.txt")
    with open("../tests/parsed_logs.json", "w", encoding="utf-8") as f:
        json.dump(_parsed_logs, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parsed_logs = parse_log_file("../data/train.log")
    with open("../data/train.json", "w", encoding="utf-8") as f:
        json.dump(parsed_logs, f, ensure_ascii=False, indent=2)
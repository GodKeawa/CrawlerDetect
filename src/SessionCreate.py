import json
import ijson
import time
from datetime import datetime, timedelta
from collections import defaultdict


def parse_timestamp(timestamp_str):
    """解析日志中的时间戳为datetime对象"""
    try:
        # 处理ISO格式时间戳
        return datetime.fromisoformat(timestamp_str)
    except ValueError:
        # 尝试其他常见时间格式
        formats = [
            "%d/%b/%Y:%H:%M:%S %z",  # 常见的Apache/Nginx格式
            "%Y-%m-%d %H:%M:%S",  # 标准日期时间格式
            "%Y/%m/%d %H:%M:%S"  # 替代日期时间格式
        ]

        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue

        # 如果所有格式都失败，引发异常
        raise ValueError(f"无法解析时间戳: {timestamp_str}")


def create_sessions(logs, session_timeout_minutes=30):
    """
    根据IP和User-Agent将日志条目分组成会话

    参数:
    - logs: JSON格式的日志条目列表
    - session_timeout_minutes: 会话超时时间（分钟）

    返回:
    - 会话列表
    """
    print(f"开始处理 {len(logs)} 条日志记录...")

    # 按IP和UA分组
    user_sessions = defaultdict(list)

    # 第一次遍历：按IP和UA分组日志条目
    for log in logs:
        # 确保必要的字段存在
        if 'ip' not in log or 'user_agent' not in log or 'timestamp' not in log:
            continue

        user_id = f"{log['ip']}_{log['user_agent']}"

        # 添加条目到相应的用户组
        user_sessions[user_id].append(log)

    print(f"已经找到 {len(user_sessions)} 个唯一的IP-UA组合")

    # 按时间戳对每组日志排序
    sessions = []
    for user_id, logs in user_sessions.items():
        # 按时间戳排序
        try:
            logs.sort(key=lambda x: parse_timestamp(x['timestamp']))
        except Exception as e:
            print(f"排序错误: {e} - 跳过用户 {user_id}")
            continue

        # 划分会话
        current_session = None
        current_start_time = None
        current_end_time = None
        current_time = None

        for log in logs:
            try:
                current_time = parse_timestamp(log['timestamp'])
                # 如果没有活跃会话或上一条记录时间超过会话超时时间，创建新会话
                if current_session is None or (current_time - current_start_time) > timedelta(
                        minutes=session_timeout_minutes):
                    # 如果有现有会话，保存它
                    if current_session is not None:
                        # 计算会话持续时间
                        current_session['duration_seconds'] = (current_end_time - current_start_time).total_seconds()
                        current_session['request_count'] = len(current_session['requests'])
                        sessions.append(current_session)

                    # 创建新会话
                    ip, user_agent = user_id.split('_', 1)  # 从user_id分离IP和UA
                    current_start_time = parse_timestamp(log['timestamp'])
                    current_end_time = current_start_time

                    current_session = {
                        'user_id': user_id,
                        'ip': ip,
                        'user_agent': user_agent,
                        'start_time': current_start_time.isoformat(),
                        'requests': [log],
                    }

                else:
                    # 更新结束时间
                    current_end_time = current_time
                    # 添加到现有会话
                    current_session['requests'].append(log)


            except Exception as e:
                print(f"处理日志时出错: {e}")
                continue

        # 保存最后一个会话
        if current_session is not None:
            current_session['duration_seconds'] = (current_end_time - current_start_time).total_seconds()
            current_session['request_count'] = len(current_session['requests'])
            sessions.append(current_session)

    print(f"共创建 {len(sessions)} 个会话")
    return sessions


def process_json(input_file, output_file, session_timeout_minutes=30):

    start_time = time.time()
    print(f"开始处理文件: {input_file}")

    # 打开输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        # 加载整个JSON数组
        try:
            objects = ijson.items(f, 'item')
            # 这个objects在这里就是相当于一个生成器，可以调用next函数取它的下一个值
            data = []
            for item in objects:
                data.append(item)
            total_records = len(data)
            print(f"加载了 {total_records} 条记录")

            sessions = create_sessions(data, session_timeout_minutes)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return

    # 保存会话数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)

    end_time = time.time()
    print(f"处理完成. 总耗时: {end_time - start_time:.2f} 秒")
    print(f"结果保存到: {output_file}")
    print(f"总共创建 {len(sessions)} 个会话")


if __name__ == "__main__":
    process_json("../data/test.json", "../data/sessions_test.json", session_timeout_minutes=30)
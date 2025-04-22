from pybrowscap.loader.csv import load_file
import crawleruseragents
import json
import ijson
import re
import os
import urllib.parse
from collections import defaultdict
from CrawlerDetect import Detect_Crawler

# 设置全局变量，减少内存消耗
browscap = load_file("../data/browscap.csv")
with open("../data/browscap.json", 'r', encoding='utf-8') as f:
    memory = json.load(f)
disallowed_paths = []
sessions = []

def detect_by_pybrowscap(user_agent: str) -> bool:
    global browscap
    global memory
    if memory.get(user_agent, -1) == -1:
        browser = browscap.search(user_agent)
        memory.update({user_agent: browser.is_crawler()})
    return memory[user_agent]

def detect_by_crawler_ua(user_agent: str) -> bool:
    return crawleruseragents.is_crawler(user_agent)


def load_robots_txt(robots_file="../data/robots.txt"):
    global disallowed_paths
    if robots_file and os.path.exists(robots_file):
        try:
            with open(robots_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"读取robots.txt文件出错: {e}")

    # 解析robots.txt
    current_agent = "*"  # 默认为所有代理

    for line in content.split('\n'):
        line = line.strip()

        if not line or line.startswith('#'):
            continue

        parts = line.split(':', 1)
        if len(parts) != 2:
            continue

        directive, value = parts[0].strip().lower(), parts[1].strip()

        if directive == 'user-agent':
            current_agent = value.lower()
        elif directive == 'disallow' and (current_agent == '*' or current_agent == 'all'):
            if value:  # 排除空规则
                disallowed_paths.append(value)


def detect_by_robots_violation(path):
    global disallowed_paths
    if not disallowed_paths:
        return False

    decoded_path = urllib.parse.unquote(path)

    for blocked_path in disallowed_paths:
        if blocked_path == '/':  # 全站禁止
            return True

        # 处理通配符 *
        if '*' in blocked_path:
            pattern = blocked_path.replace('*', '.*')
            if re.match(f"^{pattern}", decoded_path):
                return True,

        # 检查路径是否以禁止路径开头
        if decoded_path.startswith(blocked_path):
            return True


    return False

def calculate_session_request_stats():
    global sessions
    request_count_stats = defaultdict(int)

    for session in sessions:
        request_count = session.get('request_count', 0)
        request_count_stats[request_count] += 1

    # 转换为普通字典，并按请求数排序
    sorted_stats = {str(k): v for k, v in sorted(request_count_stats.items(), key=lambda x: int(x[0]))}

    return sorted_stats


def calculate_session_duration_stats():
    global sessions
    duration_brackets = {
        "0-10seconds": 0,
        "10-30seconds": 0,
        "30-60seconds": 0,
        "1-3minutes": 0,
        "3-10minutes": 0,
        "10-30minutes": 0
    }

    for session in sessions:
        duration : int = int(session.get('duration_seconds', 0))
        session['duration_seconds'] = duration

        if duration <= 10:
            duration_brackets["0-10seconds"] += 1
        elif duration <= 30:
            duration_brackets["10-30seconds"] += 1
        elif duration <= 60:
            duration_brackets["30-60seconds"] += 1
        elif duration <= 180:
            duration_brackets["1-3minutes"] += 1
        elif duration <= 600:
            duration_brackets["3-10minutes"] += 1
        else:
            duration_brackets["10-30minutes"] += 1

    return duration_brackets


def calculate_bot_stats():
    global sessions
    bot_stats = {
        "total_sessions": len(sessions),
        "bot_sessions": defaultdict(int),
        "human_sessions": defaultdict(int),
        "bot_sumup" : 0,
        "human_sumup": 0,
    }

    for session in sessions:
        bot_info = session.get('is_bot', False)
        request_count = session.get('request_count', 0)
        if bot_info:
            bot_stats['bot_sumup'] += 1
            bot_stats["bot_sessions"][request_count] += 1
        else:
            bot_stats['human_sumup'] += 1
            bot_stats["human_sessions"][request_count] += 1

    # 计算百分比
    if len(sessions) > 0:
        bot_stats["bot_percentage"] = (bot_stats["bot_sumup"] / len(sessions)) * 100
        bot_stats["human_percentage"] = (bot_stats["human_sumup"] / len(sessions)) * 100
    else:
        bot_stats["bot_percentage"] = 0
        bot_stats["human_percentage"] = 0

    return bot_stats


def detect_bots_in_sessions():
    global sessions
    counter = 0
    for session in sessions:
        counter += 1
        print(f"finished{counter}")
        user_agent = session.get('user_agent', '')
        # 综合检测结果
        bot_detection = False
        if not bot_detection:
            bot_detection = detect_by_pybrowscap(user_agent)
        if not bot_detection:
            bot_detection = detect_by_crawler_ua(user_agent)
        if not bot_detection:
            for req in session.get('requests', []):
                path = req.get('path', '')
                if not bot_detection:
                    bot_detection = detect_by_robots_violation(path)

        # 添加检测结果到会话
        session['is_bot'] = bot_detection

def process_stats(output_file: str):
    global sessions
    # 计算统计信息
    request_stats = calculate_session_request_stats()
    print("calculated!")
    duration_stats = calculate_session_duration_stats()
    print("calculated!")
    bot_stats = calculate_bot_stats()
    print("calculated!")

    # 合并统计信息
    stats = {
        "request_count_distribution": request_stats,
        "duration_distribution": duration_stats,
        "bot_statistics": bot_stats
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

def process_sessions(input_file, output_file):
    global sessions
    # 加载会话数据
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            objects = ijson.items(f, 'item')
            # 这个objects在这里就是相当于一个生成器，可以调用next函数取它的下一个值
            for item in objects:
                item["duration_seconds"] = int(item["duration_seconds"])
                sessions.append(item)
    except IOError:
        return

    # 加载检测所需数据
    load_robots_txt()

    print("loaded!")

    # 检测爬虫
    detect_bots_in_sessions()
    print("detected!")

    # save memory
    with open("../data/browscap.json", 'w', encoding='utf-8') as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)

    # 保存结果
    print("writing!")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)

def process_bots(input_file, output_file, threshold):
    global sessions
    # 加载会话数据
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            objects = ijson.items(f, 'item')
            # 这个objects在这里就是相当于一个生成器，可以调用next函数取它的下一个值
            # 检测爬虫
            for session in objects:
                if not session.get('is_bot', False):
                    result = Detect_Crawler(session, threshold)
                    session['is_bot'] = result
                sessions.append(session)
    except IOError:
        return

    # 保存结果
    print("writing!")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)


def sort_state(input_file : str):
    with open(input_file, 'r', encoding='utf-8') as f:
        stat = json.load(f)
    bot = stat["bot_statistics"]["bot_sessions"]
    stat["bot_statistics"]["bot_sessions"] = {str(k): v for k, v in sorted(bot.items(), key=lambda x: int(x[0]))}
    human = stat["bot_statistics"]["human_sessions"]
    stat["bot_statistics"]["human_sessions"] = {str(k): v for k, v in sorted(human.items(), key=lambda x: int(x[0]))}
    with open(input_file, "w") as f:
        json.dump(stat, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    process_sessions("../data/sessions_test.json", "../data/sessions_test_detected.json")
    # process_bots("../data/sessions_detected.json", "../data/sessions_temp.json", 0.9)
    # process_stats("../data/sessions_stats.json")
    # sort_state("../data/sessions_stats.json")
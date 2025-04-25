import ijson
import json
from collections import defaultdict
from CrawlerDetect import Detect_Crawler

def gen_dataset(input_file, output_file, stat_file, min_request, threshold):
    sessions = []
    request_count_stats = defaultdict(int)
    duration_brackets = {
        "0-10seconds": 0,
        "10-30seconds": 0,
        "30-60seconds": 0,
        "1-3minutes": 0,
        "3-10minutes": 0,
        "10-30minutes": 0
    }
    bot_stats = {
        "total_sessions": int(0),
        "bot_sessions": defaultdict(int),
        "human_sessions": defaultdict(int),
        "bot_sumup" : 0,
        "human_sumup": 0,
    }
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            objects = ijson.items(f, 'item')
            for session in objects:
                bot_stats['total_sessions'] += 1
                if session.get('request_count', 0) > min_request:
                    if not session.get('is_bot', False):
                        result = Detect_Crawler(session, threshold)
                        session['is_bot'] = result
                    sessions.append(session)
                # stat 统计
                request_count = session.get('request_count', 0)
                request_count_stats[request_count] += 1

                duration: int = int(session.get('duration_seconds', 0))
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

                bot_info = session.get('is_bot', False)
                request_count = session.get('request_count', 0)
                if bot_info:
                    bot_stats['bot_sumup'] += 1
                    bot_stats["bot_sessions"][request_count] += 1
                else:
                    bot_stats['human_sumup'] += 1
                    bot_stats["human_sessions"][request_count] += 1
    except IOError:
        return

    # 保存结果
    print("writing!")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)

    # 保存stats
    sorted_stats = {str(k): v for k, v in sorted(request_count_stats.items(), key=lambda x: int(x[0]))}
    if bot_stats["total_sessions"] > 0:
        bot_stats["bot_percentage"] = (bot_stats["bot_sumup"] / bot_stats["total_sessions"]) * 100
        bot_stats["human_percentage"] = (bot_stats["human_sumup"] / bot_stats["total_sessions"]) * 100
    else:
        bot_stats["bot_percentage"] = 0
        bot_stats["human_percentage"] = 0

    # 合并统计信息
    bot = bot_stats["bot_sessions"]
    bot_stats["bot_sessions"] = {str(k): v for k, v in sorted(bot.items(), key=lambda x: int(x[0]))}
    human = bot_stats["human_sessions"]
    bot_stats["human_sessions"] = {str(k): v for k, v in sorted(human.items(), key=lambda x: int(x[0]))}
    stats = {
        "request_count_distribution": sorted_stats,
        "duration_distribution": duration_brackets,
        "bot_statistics": bot_stats
    }

    with open(stat_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    pos = "../dataset"
    gen_dataset("../data/sessions_detected.json", f"{pos}/train.json", f"{pos}/stat.json", 10, 1.8)
    gen_dataset("../data/sessions_test_detected.json", f"{pos}/test.json", f"{pos}/test_stat.json", 10, 1.8)
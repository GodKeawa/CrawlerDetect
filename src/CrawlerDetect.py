import datetime
from typing import List, Dict, Any, Tuple
from collections import Counter
import re
import math
from datetime import datetime, time, timezone # 确保 timezone 也导入

# --- 示例 Session 数据 ---
session = {
    "user_id": "54.36.149.41_Mozilla/5.0 (compatible; AhrefsBot/6.1; +http://ahrefs.com/robot/)",
    "ip": "54.36.149.41",
    "user_agent": "Mozilla/5.0 (compatible; AhrefsBot/6.1; +http://ahrefs.com/robot/)",
    "start_time": "2019-01-22T03:56:14+03:30", # 注意时区信息
    "requests": [
        {
            "ip": "54.36.149.41",
            "user": None,
            "timestamp": "2019-01-22T03:56:14+03:30", # ISO 8601 格式
            "method": "GET",
            "path": "/filter/27|13%20%D9%85%DA%AF%D8%A7%D9%BE%DB%8C%DA%A9%D8%B3%D9%84,27|%DA%A9%D9%85%D8%AA%D8%B1%20%D8%A7%D8%B2%205%20%D9%85%DA%AF%D8%A7%D9%BE%DB%8C%DA%A9%D8%B3%D9%84,p53",
            "decoded_path": "/filter/27|13 مگاپیکسل,27|کمتر از 5 مگاپیکسل,p53",
            "protocol": "HTTP/1.1",
            "status": 200,
            "size": 30577,
            "referer": None,
            "user_agent": "Mozilla/5.0 (compatible; AhrefsBot/6.1; +http://ahrefs.com/robot/)"
        },
        {
            "ip": "54.36.149.41",
            "user": None,
            "timestamp": "2019-01-22T04:06:30+03:30",
            "method": "GET",
            "path": "/login/auth?forwardUri=%2Ffilter%2Fb33%2Cp47%2Crf5000000%2Crt10000000%2Cstexists",
            "decoded_path": "/login/auth?forwardUri=/filter/b33,p47,rf5000000,rt10000000,stexists",
            "protocol": "HTTP/1.1",
            "status": 200,
            "size": 33274,
            "referer": None,
            "user_agent": "Mozilla/5.0 (compatible; AhrefsBot/6.1; +http://ahrefs.com/robot/)"
        },
        {
            "ip": "54.36.149.41",
            "user": None,
            "timestamp": "2019-01-22T04:08:40+03:30",
            "method": "GET",
            "path": "/blog/cosmetic/%DA%86%DA%AF%D9%88%D9%86%D9%87-%D9%86%D8%A7%D8%AE%D9%86-%D9%87%D8%A7%DB%8C-%D8%B7%D8%A8%DB%8C%D8%B9%DB%8C-%D9%88-%D8%B2%DB%8C%D8%A8%D8%A7%DB%8C%DB%8C-%D8%AF%D8%A7%D8%B4%D8%AA%D9%87-%D8%A8%D8%A7%D8%B4/",
            "decoded_path": "/blog/cosmetic/چگونه-ناخن-های-طبیعی-و-زیبایی-داشته-باش/",
            "protocol": "HTTP/1.1",
            "status": 200,
            "size": 25815,
            "referer": None,
            "user_agent": "Mozilla/5.0 (compatible; AhrefsBot/6.1; +http://ahrefs.com/robot/)"
        }
    ],
    "duration_seconds": 746,
    "request_count": 3, # 确保这个字段与 requests 列表长度一致或能动态计算
}

# --- 辅助函数 ---
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


def calculate_intervals(requests: List[dict]) -> List[float]:
    """计算请求之间的时间间隔（秒）"""
    # (代码与您提供的版本一致)
    if len(requests) < 2: return []
    timestamps = []
    for req in requests:
        dt_object = parse_timestamp(req.get("timestamp", ""))
        if dt_object: timestamps.append(dt_object)
    if len(timestamps) < 2: return []
    timestamps.sort()
    intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
    intervals = [i for i in intervals if i > 0]
    return intervals


def calculate_head_ratio(requests: List[dict]) -> float:
    """计算HEAD请求的比例"""
    # (代码与您提供的版本一致)
    if not requests: return 0.0
    head_count = sum(1 for req in requests if req.get("method", "").upper() == "HEAD")
    return head_count / len(requests)


def calculate_night_ratio(requests: List[dict]) -> float:
    """计算夜间访问的比例"""
    # (代码与您提供的版本一致)
    if not requests: return 0.0
    night_count, valid_timestamp_count = 0, 0
    night_start_hour, night_end_hour = 22, 6
    for req in requests:
        dt_object = parse_timestamp(req.get("timestamp", ""))
        if dt_object:
            valid_timestamp_count += 1
            hour = dt_object.hour
            if hour >= night_start_hour or hour < night_end_hour: night_count += 1
    if valid_timestamp_count == 0: return 0.0
    return night_count / valid_timestamp_count


def detect_access_pattern(requests: List[dict]) -> bool:
    """检测是否存在系统化的访问模式（简化版）"""
    # (代码与您提供的版本一致)
    if len(requests) < 3: return False
    paths = [req.get("decoded_path", req.get("path", "")) for req in requests]
    paths = [p for p in paths if p]
    if len(paths) < 3: return False
    pattern_score = 0
    for i in range(len(paths) - 1):
        p1, p2 = paths[i].strip('/'), paths[i+1].strip('/')
        if p2.startswith(p1 + '/') and len(p2.split('/')) == len(p1.split('/')) + 1:
            pattern_score += 1; continue
        p1_parts, p2_parts = p1.split('/'), p2.split('/')
        if len(p1_parts) > 1 and len(p1_parts) == len(p2_parts) and p1_parts[:-1] == p2_parts[:-1]:
            pattern_score += 0.5; continue
        try: # 数字递增检查部分修正，确保 m1, m2 存在再访问 groups()
            m1 = re.match(r"^(.*[/_-])?(\d+)$", p1)
            m2 = re.match(r"^(.*[/_-])?(\d+)$", p2)
            if m1 and m2:
                 base1, num1_str = m1.groups()
                 base2, num2_str = m2.groups()
                 # 确保 base1 和 base2 存在，或者都是 None
                 if base1 == base2 and int(num2_str) == int(num1_str) + 1:
                     pattern_score += 1; continue
        except (AttributeError, ValueError, TypeError): pass
    return pattern_score >= (len(paths) - 1) * 0.5


def calculate_path_repetition(requests: List[dict]) -> float:
    """计算路径重复访问的比例"""
    # (代码与您提供的版本一致)
    if not requests: return 0.0
    paths = [req.get("decoded_path", req.get("path", "")) for req in requests]
    paths = [p for p in paths if p]
    if not paths: return 0.0
    path_counts = Counter(paths)
    repeated_request_count = sum(count - 1 for count in path_counts.values() if count > 1)
    # 修正：避免除以零
    return repeated_request_count / len(paths) if paths else 0.0


def calculate_missing_referer_ratio(requests: List[dict]) -> float:
    """计算缺失引用页(Referer)的比例"""
    # (代码与您提供的版本一致)
    if not requests: return 0.0
    missing_referer_count = sum(1 for req in requests if not req.get("referer"))
    return missing_referer_count / len(requests)


def analyze_path_bigrams(requests: List[dict]) -> Tuple[float, Any]:
    """分析连续路径对(Bigram)的重复性。"""
    # (代码与您提供的版本一致)
    if len(requests) < 2: return 0.0, None
    paths = [req.get("decoded_path", req.get("path", "")) for req in requests]
    paths = [p for p in paths if p]
    if len(paths) < 2: return 0.0, None
    bigrams = list(zip(paths[:-1], paths[1:]))
    if not bigrams: return 0.0, None
    bigram_counts = Counter(bigrams)
    # 修正：处理 bigram_counts 为空的情况（虽然理论上 if not bigrams 已覆盖）
    if not bigram_counts: return 0.0, None
    most_common_bigram = bigram_counts.most_common(1)[0]
    most_common_count = most_common_bigram[1]
    most_common_bigram_path_pair = most_common_bigram[0]
    max_bigram_ratio = most_common_count / len(bigrams)
    return max_bigram_ratio, most_common_bigram_path_pair

# --- 新增：状态码和请求大小分析函数 (Step 2) ---
def analyze_status_codes(requests: List[dict]) -> Dict[str, float]:
    """分析 HTTP 状态码的分布。"""
    results = {'ratio_2xx': 0.0, 'ratio_3xx': 0.0, 'ratio_4xx': 0.0, 'ratio_5xx': 0.0, 'ratio_404': 0.0}
    if not requests: return results
    # 确保只处理整数类型的状态码
    status_codes = [req.get("status") for req in requests if isinstance(req.get("status"), int)]
    total_valid_codes = len(status_codes)
    if total_valid_codes == 0: return results
    count_2xx = sum(1 for code in status_codes if 200 <= code < 300)
    count_3xx = sum(1 for code in status_codes if 300 <= code < 400)
    count_4xx = sum(1 for code in status_codes if 400 <= code < 500)
    count_5xx = sum(1 for code in status_codes if 500 <= code < 600)
    count_404 = sum(1 for code in status_codes if code == 404)
    results['ratio_2xx'] = round(count_2xx / total_valid_codes, 4)
    results['ratio_3xx'] = round(count_3xx / total_valid_codes, 4)
    results['ratio_4xx'] = round(count_4xx / total_valid_codes, 4)
    results['ratio_5xx'] = round(count_5xx / total_valid_codes, 4)
    results['ratio_404'] = round(count_404 / total_valid_codes, 4)
    return results

def analyze_request_sizes(requests: List[dict]) -> Dict[str, float | None]:
    """分析请求返回内容的大小 (size 字段)。"""
    results = {'avg_size': None, 'std_dev_size': None}
    if not requests: return results
    # 确保只处理数字类型且非负的 size
    sizes = [req.get("size") for req in requests if isinstance(req.get("size"), (int, float)) and req.get("size") >= 0]
    if not sizes: return results
    avg_size = sum(sizes) / len(sizes)
    results['avg_size'] = round(avg_size, 2)
    if len(sizes) > 1:
        variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
        std_dev_size = math.sqrt(variance) if variance >= 0 else 0
        results['std_dev_size'] = round(std_dev_size, 2)
    else:
        results['std_dev_size'] = 0.0 # 只有一个请求，标准差为0
    return results

# --- 核心分析函数 (整合 Step 1 & Step 2) ---
def is_crawler(session: dict, threshold) -> Tuple[bool, Dict[str, Any]]:
    """
    通过分析会话行为判断是否为爬虫 (整合 Bigram, Status, Size)

    参数:
        session: 包含用户会话信息的字典

    返回:
        Tuple[bool, Dict]: (是否为爬虫, 详细分析结果)
    """
    analysis_results = {}
    requests = session.get("requests", [])
    request_count = len(requests)

    # 检查用户代理 (与您代码一致)
    is_obvious_crawler = session.get("is_bot", False)
    analysis_results["obvious_crawler"] = is_obvious_crawler
    if is_obvious_crawler:
        analysis_results["crawler_score"] = 1.0 # 使用您代码中的分数
        analysis_results["reason"] = "Obvious crawler User-Agent"

    # 检查数据量 (与您代码一致)
    if not requests or request_count < 3 :
        analysis_results["insufficient_data"] = True
        analysis_results["crawler_score"] = 0.0
        analysis_results["reason"] = "Insufficient request data"

    # --- 开始计算各项指标 (与您代码一致的部分) ---
    # 1. 速度
    duration_seconds = session.get("duration_seconds", 0)
    is_high_speed = False
    if duration_seconds > 1:
        requests_per_second = request_count / duration_seconds
        analysis_results["requests_per_second"] = round(requests_per_second, 4)
        is_high_speed = requests_per_second > 0.5 # 您的阈值
    elif request_count > 5: # 时间短但请求多
        is_high_speed = True
        analysis_results["requests_per_second"] = float('inf')
    else:
         analysis_results["requests_per_second"] = None
    analysis_results["high_speed"] = is_high_speed

    # 2. 间隔规律性
    intervals = calculate_intervals(requests)
    avg_interval, interval_std_dev = None, None
    is_regular_interval = False
    if intervals:
        avg_interval = sum(intervals) / len(intervals)
        interval_variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
        interval_std_dev = math.sqrt(interval_variance) if interval_variance >= 0 else 0
        analysis_results["avg_interval_seconds"] = round(avg_interval, 4)
        analysis_results["interval_std_dev"] = round(interval_std_dev, 4)
        if avg_interval > 0.05 and interval_std_dev < avg_interval * 0.2 and request_count > 5: # 您的条件
             is_regular_interval = True
    else:
        analysis_results["avg_interval_seconds"] = None
        analysis_results["interval_std_dev"] = None
    analysis_results["regular_interval"] = is_regular_interval

    # 3. HEAD 比例
    head_ratio = calculate_head_ratio(requests)
    analysis_results["head_request_ratio"] = round(head_ratio, 4)
    is_high_head_ratio = head_ratio > 0.2 # 您的阈值
    analysis_results["high_head_ratio"] = is_high_head_ratio

    # 4. 夜间比例
    night_ratio = calculate_night_ratio(requests)
    analysis_results["night_access_ratio"] = round(night_ratio, 4)
    is_high_night_ratio = night_ratio > 0.7 # 您的阈值
    analysis_results["high_night_ratio"] = is_high_night_ratio

    # 5. 系统模式 (旧)
    has_pattern = detect_access_pattern(requests)
    analysis_results["has_systematic_pattern"] = has_pattern

    # 6. 路径重复
    path_repetition = calculate_path_repetition(requests)
    analysis_results["path_repetition_ratio"] = round(path_repetition, 4)
    is_high_repetition = path_repetition > 0.3 # 您的阈值
    analysis_results["high_repetition"] = is_high_repetition

    # 7. 缺失 Referer (调整后)
    missing_referer_ratio = calculate_missing_referer_ratio(requests)
    analysis_results["missing_referer_ratio"] = round(missing_referer_ratio, 4)
    is_high_missing_referer = False
    missing_referer_ratio_adjusted = 0.0
    if request_count > 1:
        missing_referer_relevant_count = sum(1 for req in requests[1:] if not req.get("referer"))
        relevant_requests_for_referer = request_count - 1
        if relevant_requests_for_referer > 0:
             missing_referer_ratio_adjusted = missing_referer_relevant_count / relevant_requests_for_referer
        if missing_referer_ratio_adjusted > 0.8: # 您的阈值
             is_high_missing_referer = True
    analysis_results["missing_referer_ratio_adjusted"] = round(missing_referer_ratio_adjusted, 4)
    analysis_results["high_missing_referer"] = is_high_missing_referer

    # 8. Path Bigram (来自 Step 1)
    max_bigram_ratio, common_bigram = analyze_path_bigrams(requests)
    analysis_results['max_bigram_ratio'] = round(max_bigram_ratio, 4) if max_bigram_ratio is not None else None
    analysis_results['most_common_bigram'] = common_bigram
    is_high_bigram_repetition = max_bigram_ratio is not None and max_bigram_ratio > 0.4 and request_count > 5 # 您的条件
    analysis_results['high_bigram_repetition'] = is_high_bigram_repetition

    # --- 新增：计算状态码和请求大小指标 (Step 2) ---
    # 9. 状态码分析
    status_code_analysis = analyze_status_codes(requests)
    analysis_results.update(status_code_analysis) # 合并结果
    # 定义判断条件 (示例阈值，可调整)
    is_high_404_ratio = status_code_analysis.get('ratio_404', 0.0) > 0.1 and request_count > 5
    analysis_results['high_404_ratio'] = is_high_404_ratio
    is_high_4xx_ratio = status_code_analysis.get('ratio_4xx', 0.0) > 0.2 and request_count > 5
    analysis_results['high_4xx_ratio'] = is_high_4xx_ratio

    # 10. 请求大小分析
    size_analysis = analyze_request_sizes(requests)
    analysis_results.update(size_analysis) # 合并结果
    avg_size = size_analysis.get('avg_size')
    std_dev_size = size_analysis.get('std_dev_size')
    # 定义判断条件 (示例阈值，可调整)
    is_low_size_variance = False
    if avg_size is not None and avg_size > 100 and std_dev_size is not None and request_count > 5:
        if std_dev_size < avg_size * 0.1: # 标准差小于平均值的10%
            is_low_size_variance = True
    analysis_results['low_size_variance'] = is_low_size_variance
    is_low_avg_size = avg_size is not None and avg_size < 500 and request_count > 5 # 平均小于500字节
    analysis_results['low_avg_size'] = is_low_avg_size
    # --- Step 2 指标计算结束 ---


    # --- 综合判断 (加权评分 - 整合版) ---
    crawler_score = 0
    # 使用您代码中的权重，并为新特征添加权重
    # !! 注意：这些权重仅为示例，强烈建议根据实际数据调优 !!
    score_weights = {
        "high_speed": 0.6,              # 您代码中的权重
        "regular_interval": 0.9,        # 您代码中的权重
        "high_head_ratio": 0.5,         # 您代码中的权重
        "high_night_ratio": 0.3,        # 您代码中的权重
        "has_systematic_pattern": 0.4,  # 您代码中的权重
        "high_repetition": 0.5,         # 您代码中的权重
        "high_missing_referer": 0.7,    # 您代码中的权重
        "high_bigram_repetition": 0.9,  # 您代码中的权重 (来自 Step 1)

        # 为 Step 2 新特征添加示例权重
        "high_404_ratio": 0.8,          # 高 404 权重较高
        "high_4xx_ratio": 0.5,          # 4xx 权重中等
        "low_size_variance": 0.7,       # 低大小方差权重较高
        "low_avg_size": 0.4             # 低平均大小权重中等
    }

    reasons = [] # 记录触发了哪些规则
    for key, weight in score_weights.items():
        if analysis_results.get(key, False): # 使用 get 保证健壮性
            crawler_score += weight
            reasons.append(f"{key}(+{weight})")

    analysis_results["crawler_score"] = round(crawler_score, 2)
    analysis_results["reasons"] = ", ".join(reasons) if reasons else "None"

    # 判定为爬虫的阈值 - 基于您代码中的阈值 1.5，并考虑新加的分数
    # !! 这个阈值同样需要根据实际数据调优 !!
    # threshold = 1.6 # 示例：略微提高阈值，因为添加了更多评分项
    is_crawler_result = (crawler_score >= threshold) or is_obvious_crawler
    analysis_results["threshold"] = threshold

    return is_crawler_result, analysis_results

# --- 主调用函数 (与您代码一致) ---
def Detect_Crawler(session: dict, threshold) -> bool:
    """检测会话是否为爬虫的主函数"""
    is_crawler_result, analysis = is_crawler(session, threshold)
    # print(f"--- Analysis for Session ---")
    # print(json.dumps(analysis, indent=2, ensure_ascii=False))
    # print(f"Is Crawler: {is_crawler_result}")
    # print("-" * 26)
    return is_crawler_result
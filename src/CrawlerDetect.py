import json
import datetime
from typing import List, Dict, Any, Tuple
from collections import Counter
import re
import math
from datetime import datetime

session = {
    "user_id": "54.36.149.41_Mozilla/5.0 (compatible; AhrefsBot/6.1; +http://ahrefs.com/robot/)",
    "ip": "54.36.149.41",
    "user_agent": "Mozilla/5.0 (compatible; AhrefsBot/6.1; +http://ahrefs.com/robot/)",
    "start_time": "2019-01-22T03:56:14+03:30",
    "requests": [
        {
            "ip": "54.36.149.41",
            "user": None,
            "timestamp": "2019-01-22T03:56:14+03:30",
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
    "request_count": 3,
}

def is_crawler(session: dict) -> Tuple[bool, Dict[str, Any]]:
    """
    通过分析会话行为判断是否为爬虫
    
    参数:
        session: 包含用户会话信息的字典
        
    返回:
        Tuple[bool, Dict]: (是否为爬虫, 详细分析结果)
    """
    analysis_results = {}
    
    # 检查用户代理是否明显为爬虫
    is_obvious_crawler = check_obvious_crawler(session.get("user_agent", ""))
    analysis_results["obvious_crawler"] = is_obvious_crawler
    
    if is_obvious_crawler:
        return True, analysis_results
    
    # 检查请求速率和时间间隔
    requests = session.get("requests", [])
    if not requests:
        analysis_results["insufficient_data"] = True
        return False, analysis_results
    
    # 计算访问速度 (请求数/持续时间)
    duration_seconds = session.get("duration_seconds", 0)
    request_count = len(requests)
    if duration_seconds > 0:
        requests_per_second = request_count / duration_seconds
        analysis_results["requests_per_second"] = requests_per_second
        is_high_speed = requests_per_second > 0.1  # 每10秒超过1个请求
        analysis_results["high_speed"] = is_high_speed
    else:
        analysis_results["requests_per_second"] = None
        is_high_speed = False
        analysis_results["high_speed"] = False
    
    # 计算平均间隔和间隔方差
    intervals = calculate_intervals(requests)
    avg_interval, interval_variance = None, None
    if intervals:
        avg_interval = sum(intervals) / len(intervals)
        interval_variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
        analysis_results["avg_interval_seconds"] = avg_interval
        analysis_results["interval_variance"] = interval_variance
        is_regular_interval = interval_variance < avg_interval * 0.5  # 方差小于平均值的一半
        analysis_results["regular_interval"] = is_regular_interval
    else:
        analysis_results["regular_interval"] = False
    
    # 计算HEAD请求占比
    head_ratio = calculate_head_ratio(requests)
    analysis_results["head_request_ratio"] = head_ratio
    is_high_head_ratio = head_ratio > 0.2  # HEAD请求超过20%
    analysis_results["high_head_ratio"] = is_high_head_ratio
    
    # 计算夜晚访问占比
    night_ratio = calculate_night_ratio(requests)
    analysis_results["night_access_ratio"] = night_ratio
    is_high_night_ratio = night_ratio > 0.5  # 夜间访问超过50%
    analysis_results["high_night_ratio"] = is_high_night_ratio
    
    # 检测访问模式 (DFS/BFS)
    has_pattern = detect_access_pattern(requests)
    analysis_results["has_systematic_pattern"] = has_pattern
    
    # 检测重复访问比例
    path_repetition = calculate_path_repetition(requests)
    analysis_results["path_repetition_ratio"] = path_repetition
    is_high_repetition = path_repetition > 0.2  # 重复访问超过20%
    analysis_results["high_repetition"] = is_high_repetition
    
    # 检测缺失引用页比例
    missing_referer_ratio = calculate_missing_referer_ratio(requests)
    analysis_results["missing_referer_ratio"] = missing_referer_ratio
    is_high_missing_referer = missing_referer_ratio > 0.7  # 缺失引用页超过70%
    analysis_results["high_missing_referer"] = is_high_missing_referer
    
    # 综合判断
    # 使用权重法对各项指标进行加权计算
    crawler_score = 0
    score_weights = {
        "obvious_crawler": 1.0,
        "high_speed": 0.7,
        "regular_interval": 0.6,
        "high_head_ratio": 0.5,
        "high_night_ratio": 0.3,
        "has_systematic_pattern": 0.8,
        "high_repetition": 0.4,
        "high_missing_referer": 0.6
    }
    
    for key, weight in score_weights.items():
        if analysis_results.get(key, False):
            crawler_score += weight
    
    analysis_results["crawler_score"] = crawler_score
    
    # 判定为爬虫的阈值
    is_crawler_result = crawler_score >= 1.0
    
    return is_crawler_result, analysis_results


def check_obvious_crawler(user_agent: str) -> bool:
    """检查用户代理是否明显为爬虫"""
    crawler_keywords = [
        "bot", "crawler", "spider", "scraper", "ahrefsbot", "baiduspider", "googlebot",
        "yandexbot", "bingbot", "slurp", "duckduckbot", "selenium", "phantomjs", "headless"
    ]
    
    user_agent_lower = user_agent.lower()
    return any(keyword in user_agent_lower for keyword in crawler_keywords)


def calculate_intervals(requests: List[dict]) -> List[float]:
    """计算请求之间的时间间隔（秒）"""
    if len(requests) < 2:
        return []
    
    timestamps = []
    for req in requests:
        try:
            ts = datetime.strptime(req.get("timestamp", ""), "%Y-%m-%dT%H:%M:%S%z")
            timestamps.append(ts)
        except (ValueError, TypeError):
            continue
    
    if len(timestamps) < 2:
        return []
    
    # 确保时间戳是按顺序排列的
    timestamps.sort()
    
    # 计算连续请求之间的间隔（秒）
    intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                 for i in range(len(timestamps)-1)]
    
    return intervals


def calculate_head_ratio(requests: List[dict]) -> float:
    """计算HEAD请求的比例"""
    if not requests:
        return 0.0
    
    head_count = sum(1 for req in requests if req.get("method") == "HEAD")
    return head_count / len(requests)


def calculate_night_ratio(requests: List[dict]) -> float:
    """计算夜间访问的比例（22:00-06:00）"""
    if not requests:
        return 0.0
    
    night_count = 0
    total_count = 0
    
    for req in requests:
        try:
            ts = datetime.strptime(req.get("timestamp", ""), "%Y-%m-%dT%H:%M:%S%z")
            hour = ts.hour
            if hour >= 22 or hour < 6:
                night_count += 1
            total_count += 1
        except (ValueError, TypeError):
            continue
    
    if total_count == 0:
        return 0.0
    
    return night_count / total_count


def detect_access_pattern(requests: List[dict]) -> bool:
    """
    检测是否存在系统化的访问模式（如DFS或BFS）
    比较简化的实现，检查路径是否有层次性或系统性
    """
    if len(requests) < 3:
        return False
    
    # 提取所有URL路径
    paths = [req.get("decoded_path", req.get("path", "")) for req in requests]
    
    # 检查是否有明显的层次结构（如DFS模式）
    path_components = [p.strip('/').split('/') for p in paths if p]
    
    # 检测共同前缀
    common_prefix_count = 0
    for i in range(len(path_components) - 1):
        if i < len(path_components) - 1:
            current = path_components[i]
            next_path = path_components[i + 1]
            
            # 检查是否有共同前缀但后面部分不同（层次性访问的特征）
            if len(current) > 0 and len(next_path) > 0:
                common_len = 0
                for j in range(min(len(current), len(next_path))):
                    if current[j] == next_path[j]:
                        common_len += 1
                    else:
                        break
                
                if common_len > 0 and common_len < min(len(current), len(next_path)):
                    common_prefix_count += 1
    
    # 如果超过一半的连续请求具有共同前缀，可能是系统性访问
    return common_prefix_count >= (len(requests) - 1) * 0.5


def calculate_path_repetition(requests: List[dict]) -> float:
    """计算路径重复访问的比例"""
    if not requests:
        return 0.0
    
    paths = [req.get("decoded_path", req.get("path", "")) for req in requests]
    path_counts = Counter(paths)
    
    # 计算重复路径的请求数
    repeated_requests = sum(count - 1 for count in path_counts.values() if count > 1)
    
    return repeated_requests / len(requests)


def calculate_missing_referer_ratio(requests: List[dict]) -> float:
    """计算缺失引用页的比例"""
    if not requests:
        return 0.0
    
    missing_referer_count = sum(1 for req in requests if not req.get("referer"))
    return missing_referer_count / len(requests)


def Detect_Crawler(session: dict) -> bool:
    """
    检测会话是否为爬虫的主函数
    
    参数:
        session: 包含用户会话信息的字典
        
    返回:
        bool: 是否为爬虫
    """
    is_crawler_result, analysis = is_crawler(session)
    
    # 可选：打印详细分析结果用于调试
    # print(json.dumps(analysis, indent=2))
    
    return is_crawler_result
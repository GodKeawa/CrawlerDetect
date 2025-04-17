import json

with open("../data/sessions_stats.json", 'r', encoding='utf-8') as f:
    stat = json.load(f)

def sort_state():
    global stat
    bot = stat["bot_statistics"]["bot_sessions"]
    stat["bot_statistics"]["bot_sessions"] = {str(k): v for k, v in sorted(bot.items(), key=lambda x: int(x[0]))}
    human = stat["bot_statistics"]["human_sessions"]
    stat["bot_statistics"]["human_sessions"] = {str(k): v for k, v in sorted(human.items(), key=lambda x: int(x[0]))}
    with open("../data/sessions_stats.json", "w") as f:
        json.dump(stat, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    sort_state()
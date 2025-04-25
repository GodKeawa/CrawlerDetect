import ijson
import json
from typing import List

def ZipSet(input: str, output: str ,keys : List[str]):
    with open(input, 'r', encoding='utf-8') as f:
        objects = ijson.items(f, 'item')
        zip_set: list = []
        index : int = 0
        for item in objects:
            index += 1
            if index % 4:
                continue
            requests = item.get('requests', [])
            zip_request: list = []
            for request in requests:
                single_request: dict = {}
                for key, value in request.items():
                    if key in keys:
                        single_request[key] = value
                zip_request.append(single_request)
                if len(zip_request) == 256:
                    break
            zip_set.append({"requests" : zip_request})

    with open(output, 'w', encoding='utf-8') as f:
        json.dump(zip_set, f, ensure_ascii=False, indent=2)


def ZipAllSet(inputs: List[str], output: str, keys : List[str]):
    zip_set: list = []
    for input in inputs:
        index: int = 0
        with open(input, 'r', encoding='utf-8') as f:
            objects = ijson.items(f, 'item')
            for item in objects:
                index += 1
                if index % 4:
                    continue
                requests = item.get('requests', [])
                zip_request: list = []
                for request in requests:
                    single_request: dict = {}
                    for key, value in request.items():
                        if key in keys:
                            single_request[key] = value
                    zip_request.append(single_request)
                    if len(zip_request) == 256:
                        break
                zip_set.append({"requests": zip_request, "is_bot" : item.get("is_bot", False)})

    with open(output, 'w', encoding='utf-8') as f:
        json.dump(zip_set, f, ensure_ascii=False, indent=2)

def ZipAndBalanceSet(inputs: List[str], output: str, keys : List[str]):
    bot : int = 0
    human : int = 0
    zip_set: list = []
    for input in inputs:
        index: int = 0
        with open(input, 'r', encoding='utf-8') as f:
            objects = ijson.items(f, 'item')
            for item in objects:
                if not item["is_bot"]:
                    index += 1
                    if index % 4:
                        continue
                if item.get("is_bot", False):
                    bot += 1
                else:
                    human += 1
                requests = item.get('requests', [])
                zip_request: list = []
                for request in requests:
                    single_request: dict = {}
                    for key, value in request.items():
                        if key in keys:
                            single_request[key] = value
                    zip_request.append(single_request)
                zip_set.append({"requests": zip_request, "is_bot" : item.get("is_bot", False)})

    print(bot, human)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(zip_set, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    #ZipSet("../dataset/train.json", "../dataset/train_zip.json", ["method", "decoded_path", "status"])
    #ZipSet("../dataset/test.json", "../dataset/test_zip.json", ["method", "decoded_path", "status"])
    #ZipAllSet(["../dataset/train.json", "../dataset/test.json"], "../dataset/all_zip.json", ["method", "decoded_path", "status", "size", "timestamp"])
    ZipAndBalanceSet(["../dataset/train.json", "../dataset/test.json"], "../dataset/all_zip.json", ["method", "decoded_path", "status", "size", "timestamp"])
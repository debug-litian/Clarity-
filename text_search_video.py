import faiss
import pickle
import torch
import clip
import csv
import time
import os
import numpy as np
from datetime import datetime
from video_player import play_video_seek

# 全局路径配置
DEVICE = "cpu"
INDEX_PATH = "./video_index.faiss"
MAPPING_PATH = "./frame_mapping.pkl"
REPORT_DIR = "./reports"

# 加载CLIP模型
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

def text_search(text_query, top_k=10):
    """文本检索，返回结构化结果+检索耗时"""
    index = faiss.read_index(INDEX_PATH)
    with open(MAPPING_PATH, "rb") as f:
        frame_mapping = pickle.load(f)
    start_time = time.time()
    text_token = clip.tokenize([text_query]).to(DEVICE)
    with torch.no_grad():
        text_feat = model.encode_text(text_token)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
    text_np = text_feat.cpu().numpy().astype(np.float32)
    dists, ids = index.search(text_np, top_k)
    cost_sec = round(time.time() - start_time, 2)
    res_list = []
    for score, idx in zip(dists[0], ids[0]):
        ch, vid_path, ts = frame_mapping[idx]
        ffplay_cmd = f'ffplay "{vid_path}" -ss {ts} -autoexit'
        res_list.append({
            "query": text_query,
            "cost_sec": cost_sec,
            "score": round(1 - score/2, 4),
            "channel": ch,
            "video_path": vid_path,
            "timestamp_sec": round(ts, 2),
            "ffplay_cmd": ffplay_cmd,
            "is_false_alarm": ""
        })
    return res_list, cost_sec

def export_csv_report(result_list):
    """导出检索结果为CSV测试报告"""
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(REPORT_DIR, f"search_report_{timestamp}.csv")
    headers = [
        "检索关键词", "检索耗时(秒)", "匹配得分", "所属通道",
        "录像文件路径", "帧时间戳(秒)", "ffplay跳转命令", "是否误报录像"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in result_list:
            writer.writerow({
                "检索关键词": row["query"],
                "检索耗时(秒)": row["cost_sec"],
                "匹配得分": row["score"],
                "所属通道": row["channel"],
                "录像文件路径": row["video_path"],
                "帧时间戳(秒)": row["timestamp_sec"],
                "ffplay跳转命令": row["ffplay_cmd"],
                "是否误报录像": row["is_false_alarm"]
            })
    return csv_path

# 命令行单独检索入口
if __name__ == "__main__":
    word = input("请输入检索关键词：")
    res, cost = text_search(word, top_k=10)
    print(f"检索耗时：{cost} 秒，共匹配 {len(res)} 条结果")
    export_csv_report(res)
    print("CSV报告已生成至 ./reports 文件夹")
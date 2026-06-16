import os
import cv2
import torch
import clip
import faiss
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image  # 新增PIL导入，修复numpy数组报错

# 全局路径配置
SAMPLE_FPS = 1
DEVICE = "cpu"
VIDEO_DIR = "./nvr_video"
INDEX_PATH = "./video_index.faiss"
MAPPING_PATH = "./frame_mapping.pkl"

# 加载CLIP模型
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

def scan_all_channel_video(root_dir):
    """全盘递归遍历，自动识别所有ch开头通道文件夹"""
    video_list = []
    for dirpath, _, filenames in os.walk(root_dir):
        channel = os.path.basename(dirpath)
        if not channel.startswith("ch"):
            continue
        for fname in filenames:
            if fname.lower().endswith((".mp4", ".avi")):
                full_path = os.path.join(dirpath, fname)
                video_list.append((channel, full_path))
    return video_list

def extract_video_frame_feature(video_path, sample_fps):
    """单视频抽帧+CLIP特征提取（修复PIL/numpy类型报错）"""
    cap = cv2.VideoCapture(video_path)
    origin_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(origin_fps / sample_fps)
    frame_feats = []
    timestamps = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            ts = frame_idx / origin_fps
            timestamps.append(ts)
            # 修复1：正确cv2颜色转换常量
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 修复2：numpy数组转为PIL Image，解决img should be PIL Image报错
            pil_img = Image.fromarray(rgb)
            img_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = model.encode_image(img_tensor)
                feat /= feat.norm(dim=-1, keepdim=True)
            frame_feats.append(feat.cpu().numpy())
        frame_idx += 1
    cap.release()
    if not frame_feats:
        return None, None
    feats_np = np.vstack(frame_feats).astype(np.float32)
    return feats_np, timestamps

def build_batch_index():
    """批量一次性处理全部识别到的通道录像，构建索引"""
    video_list = scan_all_channel_video(VIDEO_DIR)
    all_feats = []
    frame_mapping = []
    print(f"全盘扫描完成，共找到 {len(video_list)} 个录像文件")

    for ch, vid_path in tqdm(video_list, desc="批量处理多通道NVR录像"):
        feats, ts_list = extract_video_frame_feature(vid_path, SAMPLE_FPS)
        if feats is None:
            print(f"跳过损坏视频: {vid_path}")
            continue
        all_feats.append(feats)
        for ts in ts_list:
            frame_mapping.append((ch, vid_path, ts))

    total_feat = np.vstack(all_feats)
    index = faiss.IndexFlatL2(total_feat.shape[1])
    index.add(total_feat)
    faiss.write_index(index, INDEX_PATH)
    with open(MAPPING_PATH, "wb") as f:
        pickle.dump(frame_mapping, f)
    print(f"索引构建完成，总帧数：{len(frame_mapping)}")
    return len(frame_mapping)

if __name__ == "__main__":
    build_batch_index()
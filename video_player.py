import subprocess
# 直接把ffplay完整路径写死在默认参数里
def play_video_seek(vid_path, ts_sec, ffplay_exe="D:\ffmpeg-8.1.1-full_build\bin\ffplay.exe"):
    """调用ffplay跳转到指定时间播放录像片段"""
    cmd = [
        ffplay_exe,
        vid_path,
        "-ss", str(ts_sec),
        "-autoexit",
        "-x", "1280", "-y", "720"
    ]
    # 后台拉起播放器，不阻塞主程序
    subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW)
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import threading
from video_feat_extract import build_batch_index
from text_search_video import text_search, export_csv_report
from video_player import play_video_seek
import os

class ClipSearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NVR多通道文搜视频测试工具（误报录像核验）")
        self.root.geometry("1000x700")
        self.ffplay_path = tk.StringVar(value="D:\ffmpeg-8.1.1-full_build\bin\ffplay.exe")
        self.top_k = tk.IntVar(value=10)
        self.search_text = tk.StringVar()
        self.search_result_cache = []
        self.init_ui()

    def init_ui(self):
        # FFplay配置区域
        frame_cfg = ttk.LabelFrame(self.root, text="工具配置")
        frame_cfg.pack(fill="x", padx=10, pady=5)
        ttk.Label(frame_cfg, text="ffplay程序路径：").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(frame_cfg, textvariable=self.ffplay_path, width=60).grid(row=0, column=1)

        # 批量构建索引区
        frame_build = ttk.LabelFrame(self.root, text="全盘递归批量构建特征索引")
        frame_build.pack(fill="x", padx=10, pady=5)
        ttk.Button(frame_build, text="一键批量处理全部通道录像", command=self.start_build_thread).pack(side="left", padx=5)
        self.build_progress = ttk.Label(frame_build, text="等待执行...")
        self.build_progress.pack(side="left", padx=20)

        # 文搜检索区域
        frame_search = ttk.LabelFrame(self.root, text="文搜视频检索测试")
        frame_search.pack(fill="x", padx=10, pady=5)
        ttk.Label(frame_search, text="检索关键词：").grid(row=0, column=0, padx=5)
        ttk.Entry(frame_search, textvariable=self.search_text, width=40).grid(row=0, column=1)
        ttk.Label(frame_search, text="返回TOP_K条数：").grid(row=0, column=2, padx=5)
        ttk.Entry(frame_search, textvariable=self.top_k, width=5).grid(row=0, column=3)
        ttk.Button(frame_search, text="执行检索", command=self.do_search).grid(row=0, column=4, padx=10)
        ttk.Button(frame_search, text="导出CSV测试报告", command=self.export_all_csv).grid(row=0, column=5)

        # 检索结果列表
        frame_res = ttk.LabelFrame(self.root, text="检索匹配结果（双击行一键播放录像画面）")
        frame_res.pack(fill="both", expand=True, padx=10, pady=5)
        cols = ("score", "channel", "video", "timestamp", "cmd")
        self.tree = ttk.Treeview(frame_res, columns=cols, show="headings")
        self.tree.heading("score", text="匹配分数")
        self.tree.heading("channel", text="通道号")
        self.tree.heading("video", text="录像文件路径")
        self.tree.heading("timestamp", text="时间戳(秒)")
        self.tree.heading("cmd", text="FFplay播放指令")
        self.tree.column("video", width=400)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", self.on_double_play)

        # 运行日志输出框
        frame_log = ttk.LabelFrame(self.root, text="运行日志")
        frame_log.pack(fill="x", padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(frame_log, height=8)
        self.log_text.pack(fill="x")

    def log(self, msg):
        self.log_text.insert(tk.END, f"{msg}\n")
        self.log_text.see(tk.END)
        self.root.update()

    def start_build_thread(self):
        # 新开子线程执行批量处理，防止GUI卡死
        t = threading.Thread(target=self.run_build_index, daemon=True)
        t.start()

    def run_build_index(self):
        self.log("===== 开始全盘递归扫描所有ch通道，批量提取录像特征 =====")
        try:
            total_frame = build_batch_index()
            self.build_progress.config(text=f"索引构建完成，总抽帧数：{total_frame}")
            self.log("全部通道录像批量处理完成！")
        except Exception as e:
            err_msg = f"批量处理录像失败：{str(e)}"
            self.log(err_msg)
            messagebox.showerror("执行错误", err_msg)

    def do_search(self):
        query = self.search_text.get().strip()
        if not query:
            messagebox.showwarning("输入提示", "请输入检索关键词，例如 person in white")
            return
        self.log(f"开始执行文本检索：{query}")
        try:
            res_list, cost_sec = text_search(query, top_k=self.top_k.get())
            self.log(f"检索完成，耗时 {cost_sec} 秒，匹配到 {len(res_list)} 条录像片段")
            # 清空旧列表
            for item in self.tree.get_children():
                self.tree.delete(item)
            self.search_result_cache = res_list
            # 填充表格
            for row in res_list:
                self.tree.insert("", tk.END, values=(
                    row["score"], row["channel"], row["video_path"],
                    row["timestamp_sec"], row["ffplay_cmd"]
                ))
        except Exception as e:
            err_msg = f"检索执行异常：{str(e)}"
            self.log(err_msg)
            messagebox.showerror("检索失败", err_msg)

    def on_double_play(self, event):
        # 双击列表行，自动唤起ffplay跳转到对应时间戳
        sel_item = self.tree.selection()
        if not sel_item:
            return
        row_idx = self.tree.index(sel_item[0])
        data = self.search_result_cache[row_idx]
        play_video_seek(data["video_path"], data["timestamp_sec"], self.ffplay_path.get())
        self.log(f"唤起播放器执行命令：{data['ffplay_cmd']}")

    def export_all_csv(self):
        if not self.search_result_cache:
            messagebox.showinfo("导出提示", "暂无检索结果，无法导出报告")
            return
        csv_save_path = export_csv_report(self.search_result_cache)
        self.log(f"CSV测试报告导出成功，路径：{csv_save_path}")
        messagebox.showinfo("导出完成", f"测试报告已保存至：{csv_save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ClipSearchGUI(root)
    root.mainloop()
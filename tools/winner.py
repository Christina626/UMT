from tkinter import ttk
import tkinter as tk
import os
from tkinter.filedialog import askopenfilename
from tkinter import *
import camera as camera
import cv2
from PIL import Image, ImageTk
from tkinter.scrolledtext import ScrolledText
import queue
import logging
import time
import threading
import datetime

global vpath 
 
logger = logging.getLogger(__name__)

class Clock(threading.Thread):
    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()

    def run(self):
        global vpath
        with open("E:/code/UMT/winner.txt","w") as f:
            f.write(vpath)
        # os.system('python tools/launch.py configs/tvsum/umt_small_500e_tvsum_bk.py --checkpoint work_dirs/umt_small_500e_tvsum_bk/epoch_500.pth --eval')
        # # quit()
        # level = logging.INFO
        # logger.log(level,)
        # # for num in range(0,1000):
        # #     logger.log(level, num)
        # #     time.sleep(0.2)
        # os.system("python datasets/utils/static.py {}".format(vpath))
        print(vpath)
        a=os.popen('python tools/launch.py configs/tvsum/umt_small_500e_tvsum_bk.py --checkpoint work_dirs/umt_small_500e_tvsum_bk/epoch_500.pth --eval')
        # level = logging.INFO

        context = a.read()
        list1=context.splitlines()
        print(list1[-1])
        level = logging.INFO
        logger.log(level,list1[-1])
        # for line in context.splitlines():

        #     print(line)
        # a.close()
        # for i in a:
        #     # print(i.replace("\n",""))
        #     print(i)
        #     logger.log(level,i)
        
    def stop(self):
        self._stop_event.set()

# 登录界面
class LoginUi:
    def __init__(self, frame):
        self.frame = frame
        ttk.Label(self.frame, text='用户名').grid(column=1, row=1, columnspan=2)
        ttk.Entry(self.frame, ).grid(column=3, row=1, columnspan=3)
        ttk.Label(self.frame, text='密码').grid(column=1, row=2, columnspan=2)
        ttk.Entry(self.frame, show='*').grid(column=3, row=2, columnspan=3)
        ttk.Button(self.frame, text='注册', command=self.reg).grid(column=2, row=3, columnspan=2, pady=15)
        ttk.Button(self.frame, text='登录', command=self.cert).grid(column=4, row=3, pady=15)
    def reg(self):
        reg_top = tk.Toplevel(self.frame)
        tk.Label(reg_top, text='用户注册').grid(column=2, row=2)
    def cert(self):
        self.frame.destroy()  # 直接销毁

# 视频界面
class VideoUi:
    def __init__(self, frame):
        self.frame = frame
        self.background_image = tk.PhotoImage(file='tools/preview.png', width=540, height=320)
        self.movieLabel = ttk.Label(self.frame, image=self.background_image)
        self.movieLabel.pack()
        self.movieLabel.grid(column=1, row=1)
        ttk.Button(self.frame, text="选择文件", command=self.selectPath).grid(column=1, row=2)
        ttk.Button(self.frame, text="生成高光时刻", command=self.generate).grid(column=1, row=3)
    def selectPath(self):
        global vpath

        video_path = askopenfilename()

        if video_path[-3:] == "mp4":
            video = cv2.VideoCapture(video_path)

            vpath=video_path
            File_path = os.path.basename(vpath)
            print('path={},File_path={}'.format(vpath, File_path))
            vpath=File_path.split ( "." ) [ -2 ]
             

            while video.isOpened():
                ret, frame = video.read()  # 读取照片
                # print('读取成功')
                if ret == True:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # 转换颜色使播放时保持原有色彩
                    current_image = Image.fromarray(img).resize((540, 320))  # 将图像转换成Image对象
                    imgtk = ImageTk.PhotoImage(image=current_image)
                    self.movieLabel.imgtk = imgtk
                    self.movieLabel.config(image=imgtk)
                    self.movieLabel.update()


    def generate(self):
        self.clock = Clock()
        self.clock.start()


class QueueHandler(logging.Handler):
    """Class to send logging records to a queue
    It can be used from different threads
    The ConsoleUi class polls this queue to display records in a ScrolledText widget
    """
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)

# 日志界面
class ConsoleUi:
    def __init__(self, frame):
        self.frame = frame
        # Create a ScrolledText wdiget
        self.scrolled_text = ScrolledText(frame, state='disabled', height=12)
        self.scrolled_text.grid(row=0, column=0, sticky=(N, S, W, E))
        self.scrolled_text.configure(font='TkFixedFont')
        self.scrolled_text.tag_config('INFO', foreground='black')
        # Create a logging handler using a queue
        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        self.queue_handler.setFormatter(formatter)
        logger.addHandler(self.queue_handler)
        # Start polling messages from the queue
        self.frame.after(100, self.poll_log_queue)
    def display(self, record):
        msg = self.queue_handler.format(record)
        self.scrolled_text.configure(state='normal')
        self.scrolled_text.insert(tk.END, msg + '\n', record.levelname)
        self.scrolled_text.configure(state='disabled')
        # Autoscroll to the bottom
        self.scrolled_text.yview(tk.END)

    def poll_log_queue(self):
        # Check every 100ms if there is a new message in the queue to display
        while True:
            try:
                record = self.log_queue.get(block=False)
            except queue.Empty:
                break
            else:
                self.display(record)
        self.frame.after(100, self.poll_log_queue)



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    root = tk.Tk()
    root.title("视频高光检测系统")

    login_frame = tk.Frame(root)
    login_frame.grid(padx=140, pady=140)
    video_frame= tk.Frame(root)
    console_frame= tk.Frame(root)
    video_frame.grid(row=0,column=0)
    console_frame.grid(row=1,column=0)

    LoginUi(login_frame)
    try:
        root.wait_window(window=login_frame)  # 等待直到login销毁，不销毁后面的语句就不执行
        VideoUi(video_frame)
        ConsoleUi(console_frame)
    except:
        pass
    root.mainloop()



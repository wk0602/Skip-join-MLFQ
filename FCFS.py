import time
import threading
import numpy as np
import queue
import csv
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from collections import deque

lock = threading.Lock() # 创建锁

JOB_NUM = 99  # 发送请求的个数
nownum = 0

# 在opt-1.3B上的实验数据 单位: ms
#大概表示了Figure 4的图像
x = [1, 4, 16, 64, 256, 512, 1024]
first_time = [5.88, 5.93, 6.57, 8.04, 23.8, 43.9, 98.5]
next_time = [5.13, 5.11, 5.16, 5.22, 5.52, 5.72, 5.82]

# 通过实验数据拟合每次迭代推理时间
z1 = np.polyfit(x, first_time, 1)
p1 = np.poly1d(z1)

z2 = np.polyfit(x, next_time, 1)
p2 = np.poly1d(z2)

request_queue = queue.Queue() # 创建请求队列
FCFS_queue = deque() # 创建FCFS队列

class Request:  # 推理请求，理论上输出长度未知，但为仿真实验，需要事先确定
    def __init__(self, j_id, prompt_length, output_length):
        self.j_id = j_id # 推理请求的id，唯一的标识每一个请求
        self.prompt_length = int(prompt_length) # 输入长度，int用于确保输入了整数
        self.output_length = int(output_length) # 输出长度，int用于确保输入了整数
        self.first_iter_time = p1(self.prompt_length) # 第一次迭代的推理时间,基于上面拟合的函数模型得出
        self.next_iter_time  = p2(self.prompt_length) # 之后每次迭代的推理时间,同样基于上面拟合的函数模型得出
        self.iter_count = 0 # 请求执行了几次迭代，iter_count==output_length时完成整个推理   
        
        self.create_time = time.time()  # 请求创建时间
        
class RequestGenerator(threading.Thread): # 用户线程，继承自threading.Thread类

    def __init__(self, arrival_rate):
        super().__init__()
        self.arrival_rate = arrival_rate  # 请求发送速率，arrival rate = 1s / job interval
        
    def run(self): # 重写run方法
        prompt_length_list = []
        output_length_list = []
        
        # 此处为读取orca数据集中的数据来构造request，可自行修改路径
        f = open('Resource/Orca数据集.csv', 'r')
        with f:
            reader = csv.reader(f)
            count = 0
            for row in reader:
                if count == 0:
                    count += 1
                    continue
        # 将数据集中每行的第一个元素作为输入长度，第二个元素作为输出长度
                prompt_length_list.append(row[0])
                output_length_list.append(row[1])
                
        j_id = 0
        global nownum

        while j_id < JOB_NUM:
            output_ = output_length_list[j_id]
            input_ = prompt_length_list[j_id]
            request = Request(j_id, input_, output_)
            request_queue.put(request)
            j_id += 1
            nownum += 1
        
            time.sleep(1 / self.arrival_rate) # 按照arrival rate控制请求发送速率


# Define class
class FCFSScheduler:
    def __init__(self):
        self.executed = 0  # 已经完成的请求数量
        self.ave_jct = {} # 存储每个请求的JCT

    def getNewRequest(self, request: Request):
        with lock:
            FCFS_queue.append(request) # 将请求放入FCFS队列中
    
    def getInferenceJob(self):
        with lock:
            if len(FCFS_queue) == 0:
                return 0
            else:
                return FCFS_queue.popleft() # 获取FCFS队列中的队首请求

# 推理线程
def run(scheduler):
    while scheduler.executed != JOB_NUM: # 挨个请求执行直到所有请求都完成推理  
        if request_queue.empty() and nownum != 99: # 如果请求队列为空，等待0.1s
            time.sleep(0.01)
            continue 
        for i in range(request_queue.qsize()):
            req = request_queue.get() # 获取请求
            scheduler.getNewRequest(req) # 将请求放入调度器中
        
        job = scheduler.getInferenceJob()
        
        if job == 0:
            continue

        if job.iter_count == 0: # 第一次迭代
                iter_time = job.first_iter_time # 获取第一次迭代的推理时间
        else:
            iter_time = job.next_iter_time # 获取之后每次迭代的推理时间

        args = [iter_time, job, scheduler] # 将参数打包
        # 调用模拟推理线程
        thread_pool = ThreadPoolExecutor(max_workers=3)
        temp_thread = thread_pool.submit(lambda p: simulate_forward(*p), args)

def simulate_forward(iteration_time, job, scheduler):
    with lock:
        if job.iter_count == 0: # 第一次迭代
            time.sleep(iteration_time / 1000)  # ms
            job.iter_count += 1
            with open('FCFS推理过程.txt', 'a') as file:
                content = "job id: %d, iter: %d" % (job.j_id, job.iter_count)
                file.write(content)
                file.write('\n')
            FCFS_queue.appendleft(job) # 塞回去继续推理
            
            return 0

        for i in range(job.output_length - 1): # 模拟推理
            time.sleep(iteration_time / 1000)  # ms
            job.iter_count += 1 # 迭代次数加一
            with open('FCFS推理过程.txt', 'a') as file:
                content = "job id: %d, iter: %d" % (job.j_id, job.iter_count)
                file.write(content)
                file.write('\n')

        jct = time.time() - job.create_time # 计算jct               
        scheduler.ave_jct[job.j_id] = jct # 将jct放入调度器的jct存储字典中
        
        scheduler.executed += 1 # 已经完成的请求数量加一

if __name__ == '__main__':
    # 定义并启动发送请求的用户线程
    arrival_rate = 10
    generator = RequestGenerator(arrival_rate = arrival_rate)
    generator.start() # 启动用户线程
    
    # 定义并启动调度器线程
    scheduler = FCFSScheduler()
    run(scheduler)
    
    # 输出每个请求的jct
    
    for index in range(JOB_NUM):
        print("job id: %d, jct: %f" % (index, scheduler.ave_jct[index]))
    
    # 计算并输出平均jct
    total_jct = sum(scheduler.ave_jct.values())
    average_jct = total_jct / JOB_NUM
    print("Average JCT: %f" % average_jct)

    # 绘制jct分布图
    job_ids = range(JOB_NUM)
    jct_values = [scheduler.ave_jct[index] for index in job_ids]

    plt.figure(figsize=(10, 5))
    plt.bar(job_ids, jct_values, color='skyblue')
    plt.xlabel('Job ID')
    plt.ylabel('JCT (seconds)')
    plt.title('Job Completion Time (JCT) for Each Job    Average JCT: {:.2f}   arrival_rate: {:.2f}'.format(average_jct, arrival_rate))
    plt.show()


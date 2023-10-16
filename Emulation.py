import time
import threading
import numpy as np
import queue

JOB_NUM = 100  # 发送请求的个数

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

class Request:  # 推理请求，理论上输出长度未知，但为仿真实验，需要事先确定
    def __init__(self, j_id, prompt_length, output_length):
        self.j_id = j_id # 推理请求的id，唯一的标识每一个请求
        self.prompt_length = int(prompt_length) # 输入长度，int用于确保输入了整数
        self.output_length = int(output_length) # 输出长度，int用于确保输入了整数
        self.first_iter_time = fit_first_iter_time(prompt_length) # 第一次迭代的推理时间,基于上面拟合的函数模型得出
        self.next_iter_time  = fit_next_iter_time(prompt_length) # 之后每次迭代的推理时间,同样基于上面拟合的函数模型得出
        self.iter_count = 0 # 请求执行了几次迭代，iter_count==output_length时完成整个推理   
        self.priority = -1  # 请求目前处于第几级队列
        
        self.create_time = time.time()  # 请求创建时间
        
class RequestGenerator(threading.Thread): # 用户线程，继承自threading.Thread类

    def __init__(self, arrival_rate):
        super().__init__()
        self.arrival_rate = arrival_rate  # 请求发送速率，arrival rate = 1s / job interval
        
    def run(self): # 重写run方法
        prompt_length_list = []
        output_length_list = []
        
        # 此处为读取orca数据集中的数据来构造request，可自行修改路径
        f = open('/simulation/orca_100k.csv', 'r')
        with f:
            reader = csv.reader(f)
            for row in reader:
                if count == 0:
                    count += 1
                    continue
        # 将数据集中每行的第一个元素作为输入长度，第二个元素作为输出长度
                prompt_length_list.append(row[0])
                output_length_list.append(row[1])
                
        j_id = 0

        while j_id < JOB_NUM:
            output_ = output_length_list[j_id]
            input_ = prompt_length_list[j_id]
            request = Request(j_id, input_, output_)
            request_queue.put(request)

            j_id += 1
            
            time.sleep(1 / self.arrival_rate) # 按照arrival rate控制请求发送速率


# Define class
class SkipJoinMLFQScheduler:

    def __init__(self, first_quantum=6, quantum_rate=4, queue_num=4):
        # super().__init__() ？
        self.quantum_list = [] # 每个队列的时间片大小序列
        self.multi_level_priority_queue = [] # 用列表存储多级队列
        self.executed = 0  # 已经完成的请求数量

        # first quantum/Q1 is the min iteration time
        for i in range(queue_num):
            self.quantum_list.append(quantum_rate ** i) # 每个队列的时间片大小为前一个队列的时间片大小的quantum_rate倍
            temp_q = queue.Queue(-1) # 创建queue_num个队列，表示不同优先级
            self.multi_level_priority_queue.append(temp_q) # 将创建的队列放入多级队列列表中
            
        self.ave_jct = [] # 存储每个请求的JCT

    def getNewRequest(self, request: Request):
        # Todo: 处理缓冲区中新到达的request，根据他们的输入长度放入多级队列中
        first_iter_time = request.first_iter_time # 获取该请求第一次迭代的推理时间

        # 根据第一次迭代的推理时间确定优先级
        queue_index = 0 # 初始化
        for i in range(len(self.quantum_list)):
            if first_iter_time > self.quantum_list[i]: # 这个队列放不下
                queue_index = queue_index + 1 # 放入下一级队列
            else: # 这个队列可以放下
                break # 跳出循环
    
    def demoteRequest(self, job):
        # Todo: 将完成了推理但还没生成完毕的请求放入下一级队列
        pass
    
    def getInferenceJob(self):
        # Todo: 返回在最高优先级的队列中的队首请求
        pass
        
# 推理线程
def run(scheduler):
    while scheduler.executed != JOB_NUM:
        for i in range(request_queue.qsize())
            req = request_queue.get()
            scheduler.getNewRequest(req)

        job = scheduler.getInferenceJob()

        if job.iter_count == 0:
            iter_time = job.first_iter_time
        else:
            iter_time = job.next_iter_time

        args = [iter_time, job, scheduler]
        # 调用模拟推理线程
        temp_thread = thread_pool.submit(lambda p: simulate_forward(*p), args)


def simulate_forward(iteration_time, job, scheduler):
    
    iteration_num = scheduler.quantum_list[job.priority]  # 获取当前任务在这次推理中需要执行多少轮
    
    if iteration_num >= job.output_length - job.iter_count:
        iteration_num = job.output_length - job.iter_count

        for i in range(iteration_num):
            time.sleep(iteration_time / 1000)  # ms
            job.iter_count += 1

        jct = time.time() - job.create_time                     
        scheduler.ave_jct.append(jct)
        
        scheduler.executed += 1
        
    else:
        for i in range(iteration_num):
            time.sleep(iteration_time / 1000)  # ms
            job.iter_count += 1

        scheduler.demoteRequest(job)


if __name__ == '__main__':
    # 定义并启动发送请求的用户线程
    generator = RequestGenerator(arrival_rate=arrival_rate)
    generator.start()
    
    # 定义并启动调度器线程
    scheduler = SkipJoinMLFQScheduler(first_quantum=quantum, quantum_rate=quantum_rate, queue_num=queue_num)
    run(scheduler)

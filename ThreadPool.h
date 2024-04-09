//
// Shamelessly copied from https://stackoverflow.com/questions/15752659/thread-pooling-in-c11
// May or may not have been modified
//

#ifndef BACKPROPAGATION_THREADPOOL_H
#define BACKPROPAGATION_THREADPOOL_H

#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

class ThreadPool {
public:
    void Start();

    void QueueJob(const std::function<void()> &job);

    void Stop();

    bool busy();

private:
    void ThreadLoop();

    bool should_terminate = false;           // Tells threads to stop looking for jobs
    std::mutex queue_mutex;                  // Prevents data races to the job queue
    std::condition_variable mutex_condition; // Allows threads to wait on new jobs or termination
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> jobs;
};

void ThreadPool::Start() {
    const uint32_t num_threads = std::thread::hardware_concurrency(); // Max # of threads the system supports
    for (uint32_t ii = 0; ii < num_threads; ++ii) {
        threads.emplace_back(&ThreadPool::ThreadLoop, this);
    }
}

void ThreadPool::ThreadLoop() {
    while (true) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            mutex_condition.wait(lock, [this] {
                return !jobs.empty() || should_terminate;
            });
            if (should_terminate) {
                return;
            }
            job = jobs.front();
            jobs.pop();
        }
        job();
    }
}

void ThreadPool::QueueJob(const std::function<void()> &job) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        jobs.push(job);
    }
    mutex_condition.notify_one();
}

bool ThreadPool::busy() {
    bool poolbusy;
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        poolbusy = !jobs.empty();
    }
    return poolbusy;
}

void ThreadPool::Stop() {
    while (busy()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        should_terminate = true;
    }
    mutex_condition.notify_all();
    for (std::thread &active_thread: threads) {
        active_thread.join();
    }
    threads.clear();
}

#endif //BACKPROPAGATION_THREADPOOL_H

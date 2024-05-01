//
// Shamelessly copied from https://stackoverflow.com/questions/15752659/thread-pooling-in-c11
// May or may not have been modified
//

#ifndef BACKPROPAGATION_THREADPOOL_H
#define BACKPROPAGATION_THREADPOOL_H

#include <functional>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <queue>
#include <vector>
#include <chrono>
#include <sys/types.h>

class ThreadPool {
public:
    void Start() {
        const uint num_threads = std::thread::hardware_concurrency(); // Max # of threads the system supports
        for (uint ii = 0; ii < num_threads; ++ii) {
            threads.emplace_back(&ThreadPool::ThreadLoop, this);
        }
    }

    void QueueJob(const std::function<void()> &job) {
        {
            std::unique_lock<std::mutex> const lock(queue_mutex);
            jobs.push(job);
        }
        mutex_condition.notify_one();
    }

    void Stop() {
        while (busy()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        {
            std::unique_lock<std::mutex> const lock(queue_mutex);
            should_terminate = true;
        }
        mutex_condition.notify_all();
        for (std::thread &active_thread: threads) {
            active_thread.join();
        }
        threads.clear();
    }

    bool busy() {
        bool poolbusy;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            poolbusy = !jobs.empty();
        }
        return poolbusy;
    }

private:
    void ThreadLoop() {
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

    bool should_terminate = false;           // Tells threads to stop looking for jobs
    std::mutex queue_mutex;                  // Prevents data races to the job queue
    std::condition_variable mutex_condition; // Allows threads to wait on new jobs or termination
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> jobs;
};

#endif //BACKPROPAGATION_THREADPOOL_H

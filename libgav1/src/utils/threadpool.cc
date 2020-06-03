#include "src/utils/threadpool.h"

#if defined(_MSC_VER)
#include <process.h>
#include <windows.h>
#else  // defined(_MSC_VER)
#include <pthread.h>
#endif  // defined(_MSC_VER)
#if defined(__ANDROID__) || defined(__GLIBC__)
#include <sys/types.h>
#include <unistd.h>
#endif
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <utility>

#if defined(__ANDROID__)
#include <chrono>  // NOLINT (unapproved c++11 header)
#endif

// The glibc wrapper for the gettid() system call was added in glibc 2.30.
// Emulate it for older versions of glibc.
#if defined(__GLIBC_PREREQ)
#if !__GLIBC_PREREQ(2, 30)

#include <sys/syscall.h>

static pid_t gettid() { return static_cast<pid_t>(syscall(SYS_gettid)); }

#endif
#endif  // defined(__GLIBC_PREREQ)

namespace libgav1 {

#if defined(__ANDROID__)
namespace {

using Clock = std::chrono::steady_clock;
using Duration = Clock::duration;
constexpr Duration kBusyWaitDuration =
    std::chrono::duration_cast<Duration>(std::chrono::duration<double>(2e-3));

}  // namespace
#endif  // defined(__ANDROID__)

// static
std::unique_ptr<ThreadPool> ThreadPool::Create(int num_threads) {
  return Create(/*name_prefix=*/"", num_threads);
}

// static
std::unique_ptr<ThreadPool> ThreadPool::Create(const char name_prefix[],
                                               int num_threads) {
  if (name_prefix == nullptr || num_threads <= 0) return nullptr;
  std::unique_ptr<WorkerThread*[]> threads(new (std::nothrow)
                                               WorkerThread*[num_threads]);
  if (threads == nullptr) return nullptr;
  std::unique_ptr<ThreadPool> pool(new (std::nothrow) ThreadPool(
      name_prefix, std::move(threads), num_threads));
  if (pool != nullptr && !pool->StartWorkers()) {
    pool = nullptr;
  }
  return pool;
}

ThreadPool::ThreadPool(const char name_prefix[],
                       std::unique_ptr<WorkerThread*[]> threads,
                       int num_threads)
    : threads_(std::move(threads)), num_threads_(num_threads) {
  threads_[0] = nullptr;
  assert(name_prefix != nullptr);
  const size_t name_prefix_len =
      std::min(strlen(name_prefix), sizeof(name_prefix_) - 1);
  memcpy(name_prefix_, name_prefix, name_prefix_len);
  name_prefix_[name_prefix_len] = '\0';
}

ThreadPool::~ThreadPool() { Shutdown(); }

void ThreadPool::Schedule(std::function<void()> closure) {
  LockMutex();
  if (!queue_.GrowIfNeeded()) {
    // queue_ is full and we can't grow it. Run |closure| directly.
    UnlockMutex();
    closure();
    return;
  }
  queue_.Push(std::move(closure));
  UnlockMutex();
  SignalOne();
}

int ThreadPool::num_threads() const { return num_threads_; }

// A simple implementation that mirrors the non-portable Thread.  We may
// choose to expand this in the future as a portable implementation of
// Thread, or replace it at such a time as one is implemented.
class ThreadPool::WorkerThread : public Allocable {
 public:
  // Creates and starts a thread that runs pool->WorkerFunction().
  explicit WorkerThread(ThreadPool* pool);

  // Not copyable or movable.
  WorkerThread(const WorkerThread&) = delete;
  WorkerThread& operator=(const WorkerThread&) = delete;

  // REQUIRES: Join() must have been called if Start() was called and
  // succeeded.
  ~WorkerThread() = default;

  LIBGAV1_MUST_USE_RESULT bool Start();

  // Joins with the running thread.
  void Join();

 private:
  static void* ThreadBody(void* arg);
  void SetupName();

  ThreadPool* pool_;
#if defined(_MSC_VER)
  HANDLE handle_;
#else
  pthread_t thread_;
#endif
};

ThreadPool::WorkerThread::WorkerThread(ThreadPool* pool) : pool_(pool) {}

#if defined(_MSC_VER)

bool ThreadPool::WorkerThread::Start() {
  // Since our code calls the C run-time library (CRT), use _beginthreadex
  // rather than CreateThread. Microsoft documentation says "If a thread
  // created using CreateThread calls the CRT, the CRT may terminate the
  // process in low-memory conditions."
  uintptr_t handle = _beginthreadex(
      /*security=*/nullptr, /*stack_size=*/0,
      static_cast<unsigned int(__stdcall*)(void*)>(
          [](void* arg) -> unsigned int {
            ThreadBody(arg);
            return 0;
          }),
      this, /*initflag=*/CREATE_SUSPENDED, /*thrdaddr=*/nullptr);
  if (handle == 0) return false;
  handle_ = reinterpret_cast<HANDLE>(handle);
  ResumeThread(handle_);
  return true;
}

void ThreadPool::WorkerThread::Join() {
  WaitForSingleObject(handle_, INFINITE);
  CloseHandle(handle_);
}

void ThreadPool::WorkerThread::SetupName() {
  // Not currently supported on Windows.
}

#else  // defined(_MSC_VER)

bool ThreadPool::WorkerThread::Start() {
  return pthread_create(&thread_, nullptr, ThreadBody, this) == 0;
}

void ThreadPool::WorkerThread::Join() { pthread_join(thread_, nullptr); }

void ThreadPool::WorkerThread::SetupName() {
  if (pool_->name_prefix_[0] != '\0') {
#if defined(__APPLE__)
    // Apple's version of pthread_setname_np takes one argument and operates on
    // the current thread only. Also, pthread_mach_thread_np is Apple-specific.
    // The maximum size of the |name| buffer was noted in the Chromium source
    // code and was confirmed by experiments.
    char name[64];
    mach_port_t id = pthread_mach_thread_np(pthread_self());
    int rv = snprintf(name, sizeof(name), "%s/%" PRId64, pool_->name_prefix_,
                      static_cast<int64_t>(id));
    assert(rv >= 0);
    rv = pthread_setname_np(name);
    assert(rv == 0);
    static_cast<void>(rv);
#elif defined(__ANDROID__) || defined(__GLIBC__)
    // If the |name| buffer is longer than 16 bytes, pthread_setname_np fails
    // with error 34 (ERANGE) on Android.
    char name[16];
    pid_t id = gettid();
    int rv = snprintf(name, sizeof(name), "%s/%" PRId64, pool_->name_prefix_,
                      static_cast<int64_t>(id));
    assert(rv >= 0);
    rv = pthread_setname_np(pthread_self(), name);
    assert(rv == 0);
    static_cast<void>(rv);
#endif
  }
}

#endif  // defined(_MSC_VER)

void* ThreadPool::WorkerThread::ThreadBody(void* arg) {
  auto* thread = static_cast<WorkerThread*>(arg);
  thread->SetupName();
  thread->pool_->WorkerFunction();
  return nullptr;
}

bool ThreadPool::StartWorkers() {
  if (!queue_.Init()) return false;
  for (int i = 0; i < num_threads_; ++i) {
    threads_[i] = new (std::nothrow) WorkerThread(this);
    if (threads_[i] == nullptr) return false;
    if (!threads_[i]->Start()) {
      delete threads_[i];
      threads_[i] = nullptr;
      return false;
    }
  }
  return true;
}

void ThreadPool::WorkerFunction() {
  LockMutex();
  while (true) {
    if (queue_.Empty()) {
      if (exit_threads_) {
        break;  // Queue is empty and exit was requested.
      }
#if defined(__ANDROID__)
      // On android, if we go to a conditional wait right away, the CPU governor
      // kicks in and starts shutting the cores down. So we do a very small busy
      // wait to see if we get our next job within that period. This
      // significantly improves the performance of common cases of tile parallel
      // decoding. If we don't receive a job in the busy wait time, we then go
      // to an actual conditional wait as usual.
      UnlockMutex();
      bool found_job = false;
      const auto wait_start = Clock::now();
      while (Clock::now() - wait_start < kBusyWaitDuration) {
        LockMutex();
        if (!queue_.Empty()) {
          found_job = true;
          break;
        }
        UnlockMutex();
      }
      // If |found_job| is true, we simply continue since we already hold the
      // mutex and we know for sure that the |queue_| is not empty.
      if (found_job) continue;
      // Since |found_job_| was false, the mutex is not being held at this
      // point.
      LockMutex();
      // Ensure that the queue is still empty.
      if (!queue_.Empty()) continue;
      if (exit_threads_) {
        break;  // Queue is empty and exit was requested.
      }
#endif  // defined(__ANDROID__)
      // Queue is still empty, wait for signal or broadcast.
      Wait();
    } else {
      // Take a job from the queue.
      std::function<void()> job = std::move(queue_.Front());
      queue_.Pop();

      UnlockMutex();
      // Note that it is good practice to surround this with a try/catch so
      // the thread pool doesn't go to hell if the job throws an exception.
      // This is omitted here because Google3 doesn't like exceptions.
      std::move(job)();
      job = nullptr;

      LockMutex();
    }
  }
  UnlockMutex();
}

void ThreadPool::Shutdown() {
  // Tell worker threads how to exit.
  LockMutex();
  exit_threads_ = true;
  UnlockMutex();
  SignalAll();

  // Join all workers. This will block.
  for (int i = 0; i < num_threads_; ++i) {
    if (threads_[i] == nullptr) break;
    threads_[i]->Join();
    delete threads_[i];
  }
}

}  // namespace libgav1
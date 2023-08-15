#include <iostream>

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

// constexpr size_t kKB = 1024;
constexpr size_t kMB = 1024 * 1024;
// constexpr size_t kGB = 1024 * 1024 * 1024;
// constexpr size_t kRandomSeed = 1009;
struct MyKernel {
  void operator()() const {
  }
};

int ReadWriteSpeed(sycl::queue &q,
                   sycl::buffer<char, 1> &device_buffer1,
                   sycl::buffer<char, 1> &device_buffer2,
                   char *host_buffer1,
                   char *host_buffer2,
                   size_t block_bytes,
                   size_t total_bytes
) {
  size_t num_transfers = total_bytes / block_bytes;
  assert(num_transfers > 0);

  auto start = std::chrono::steady_clock::now();

  for (size_t i = 0; i < num_transfers; i++) {

    q.submit([&](sycl::handler &h) {
      auto buf_range = block_bytes / sizeof(char);
      auto buf_offset = (i * block_bytes) / sizeof(char);
      sycl::accessor<char, 1, sycl::access::mode::write> mem1(device_buffer1, h, buf_range, buf_offset);
      h.copy(&host_buffer1[buf_offset], mem1);
    });

    q.submit([&](sycl::handler &h) {
      auto buf_range = block_bytes / sizeof(char);
      auto buf_offset = (i * block_bytes) / sizeof(char);
      sycl::accessor<char, 1, sycl::access::mode::read> mem2(device_buffer2, h, buf_range, buf_offset);
      h.copy(mem2, &host_buffer2[buf_offset]);
    });

  }
  q.wait();

  auto end = std::chrono::steady_clock::now();
  double time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

  float throughput = ((float)total_bytes / kMB) / (float)time_span;

  std::cout << throughput << " MB/s" << std::endl;
  return 0;
}

int bi_directional_speed_test(sycl::queue &q) {

  // Total bytes to transfer
  constexpr size_t kMaxBytes = 8 * kMB;  // 8 MB;
  constexpr size_t kMaxChars = kMaxBytes / sizeof(char);

  // Block size of each transfer in bytes
  // constexpr size_t kMinBytes = 32 * kKB;  // 32 KB
  // size_t block_bytes = kMinBytes;


  // int kIterations = 8;

  sycl::buffer<char, 1> device_buffer1{sycl::range<1>{kMaxChars}, {sycl::property::buffer::mem_channel{1}}};
  sycl::buffer<char, 1> device_buffer2{sycl::range<1>{kMaxChars}, {sycl::property::buffer::mem_channel{2}}};

  char *host_buffer1 = new char[kMaxBytes];
  char *host_buffer2 = new char[kMaxBytes];

  for (int i = 0; i < kMaxChars; i++) {
    host_buffer1[i] = rand();
  }


  ReadWriteSpeed(q, device_buffer1, device_buffer2, host_buffer1, host_buffer2, 8388608, 8388608);

  bool passed = true;
  for (int i = 0; i < kMaxChars; i++) {
    passed &= host_buffer1[i] == host_buffer2[i];
  }
  return passed;
  
}

int main() {
#if FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif
  sycl::property_list q_prop_list{sycl::property::queue::enable_profiling()};
  sycl::queue q(selector, fpga_tools::exception_handler, q_prop_list);

  q.single_task<class MY_KERNEL>(MyKernel{});

  bool passed = bi_directional_speed_test(q);
  std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

  return 0;
}
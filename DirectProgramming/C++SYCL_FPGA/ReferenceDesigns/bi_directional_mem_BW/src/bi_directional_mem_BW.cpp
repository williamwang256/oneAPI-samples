#include <iostream>

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "exception_handler.hpp"

int bi_directional_speed_test(sycl::queue &q) {

  constexpr size_t kNumItems = 1024;
  constexpr size_t kTransferSize = 8;
  constexpr size_t kNumTransfers = kNumItems / kTransferSize;

  int kIterations = 8;

  sycl::buffer<int, 1> devicebuffer1{sycl::range<1>{kNumItems}, {sycl::property::buffer::mem_channel{1}}};
  sycl::buffer<int, 1> devicebuffer2{sycl::range<1>{kNumItems}, {sycl::property::buffer::mem_channel{2}}};

  int *hostbuffer1 = new int[kNumItems];
  int *hostbuffer2 = new int[kNumItems];

  for (int i = 0; i < kNumItems; i++) {
    hostbuffer1[i] = rand();
  }

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < kIterations; i++) {
    for (int j = 0; j < kNumTransfers; j++) {
      q.submit([&](sycl::handler &h) {
        sycl::accessor<int, 1, sycl::access::mode::write> mem(devicebuffer1, h, kNumItems, 0);
        h.copy(hostbuffer1, mem);
      });
      q.submit([&](sycl::handler &h) {
        sycl::accessor<int, 1, sycl::access::mode::read> mem(devicebuffer1, h, kNumItems, 0);
        h.copy(mem, hostbuffer2);
      });
    }
    q.wait();
  
  }

  auto end = std::chrono::steady_clock::now();
  double time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

  std::cout << (unsigned)(time_span * 1000) << " ms" << std::endl;

  bool passed = true;
  for (int i = 0; i < kNumItems; i++) {
    passed &= hostbuffer1[i] == hostbuffer2[i];
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

  bool passed = bi_directional_speed_test(q);
  std::cout << (passed ? "PASSED" : "FAILED") << std::endl;

  return 0;
}
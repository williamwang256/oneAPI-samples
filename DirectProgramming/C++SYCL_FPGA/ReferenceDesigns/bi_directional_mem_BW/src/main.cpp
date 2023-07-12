#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <cmath>
#include <iomanip>
#include <random>

#include "exception_handler.hpp"

#if defined(FPGA_EMULATOR)
constexpr int kTimes = 20;
constexpr int kSize = 4096;
#elif defined(FPGA_SIMULATOR)
constexpr int kTimes = 10;
constexpr int kSize = 1024;
#else
constexpr int kTimes = 100;
constexpr int kSize = 2621440;
#endif

#if defined(FPGA_SIMULATOR)
constexpr int kPow = 5;
#else
constexpr int kPow = 20;
#endif

bool pass = true;

// Forward declare the kernel name in the global scope.
// This FPGA best practice reduces name mangling in the optimization reports.
class SimpleVpow;

void SimplePow(sycl::queue &q, sycl::buffer<float, 1> &buffer_a, sycl::buffer<float, 1> &buffer_b, sycl::event &e) {
  e = q.submit([&](sycl::handler &h) {
    sycl::accessor accessor_a(buffer_a, h, sycl::read_only);
    sycl::accessor accessor_b(buffer_b, h, sycl::read_write, sycl::no_init);

    const int num = kSize;
    assert(kPow >= 2);
    const int p = kPow - 1;  // Assumes pow >= 2;

    h.single_task<SimpleVpow>([=]() [[intel::kernel_args_restrict]] {
      for (int j = 0; j < p; j++) {
        if (j == 0) {
          for (int i = 0; i < num; i++) {
            accessor_b[i] = accessor_a[i] * accessor_a[i];
          }
        } else {
          for (int i = 0; i < num; i++) {
            accessor_b[i] = accessor_b[i] * accessor_a[i];
          }
        }
      }
    });
  });

  sycl::event update_host_event;
  update_host_event = q.submit([&](sycl::handler &h) {
    sycl::accessor accessor_b(buffer_b, h, sycl::read_only);
    h.update_host(accessor_b);
  });
}

unsigned long SyclGetExecTimeNs(sycl::event e) {
  unsigned long start_time = e.get_profiling_info<sycl::info::event_profiling::command_start>();
  unsigned long end_time = e.get_profiling_info<sycl::info::event_profiling::command_end>();
  return (end_time - start_time);
}

float MyPow(float input, int pow) {
  return (pow == 0) ? 1 : input * MyPow(input, pow - 1);
}

void ProcessOutput(sycl::buffer<float, 1> &input_buf, sycl::buffer<float, 1> &output_buf, sycl::event e, unsigned long &total_kernel_time_per_slot) {
  sycl::host_accessor input_buf_acc(input_buf, sycl::read_only);
  sycl::host_accessor output_buf_acc(output_buf, sycl::read_only);
  int num_errors = 0;
  int num_errors_to_print = 10;
  constexpr double epsilon = 0.01;

  for (int i = 0; i < kSize / 8; i++) {
    const double expected_value = MyPow(input_buf_acc[i], kPow);
    const bool out_invalid = std::abs((output_buf_acc[i] - expected_value) / expected_value) > epsilon;
    if ((num_errors < num_errors_to_print) && out_invalid) {
      std::cout << "error" << std::endl;
      num_errors++;
    }
  }
  total_kernel_time_per_slot += SyclGetExecTimeNs(e);
}


void ProcessInput(sycl::buffer<float, 1> &buf) {
  // sycl::host_accessor buf_acc(buf, sycl::write_only, sycl::no_init);
  // auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  // std::default_random_engine dre(seed);
  // std::uniform_real_distribution<float> di(1.0f, 2.0f);
  // float start_val = di(dre);
  // for (int i = 0; i < kSize / 8; i++) {
  //   buf_acc[i] = start_val;
  //   start_val++;
  // }
  
}

int main() {

#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  auto prop_list = sycl::property_list{sycl::property::queue::enable_profiling()};
  sycl::queue q(selector, fpga_tools::exception_handler, prop_list);

  std::vector<sycl::buffer<float, 1>> input_buf;
  std::vector<sycl::buffer<float, 1>> output_buf;

  sycl::event sycl_events[2];
  unsigned long total_kernel_time_per_slot[2];
  unsigned long total_kernel_time = 0;

  for (int i = 0; i < 2; i++) {
    input_buf.push_back(sycl::buffer<float, 1>(sycl::range<1>(kSize)));
    output_buf.push_back(sycl::buffer<float, 1>(sycl::range<1>(kSize)));
  }

  for (int i = 0; i < 2; i++) {
    total_kernel_time_per_slot[i] = 0;
  }

  auto start = std::chrono::steady_clock::now();

  // Process input for first 2 kernel launches and queue them. Then block
  // on processing the output of the first kernel.
  ProcessInput(input_buf[0]);
  ProcessInput(input_buf[1]);

  SimplePow(q, input_buf[0], output_buf[0], sycl_events[0]);

  for (int i = 1; i < kTimes; i++) {

    // Launch the next kernel
    SimplePow(q, input_buf[i % 2], output_buf[i % 2], sycl_events[i % 2]);

    // Process output from previous kernel. This will block on kernel
    // completion.
    ProcessOutput(input_buf[(i - 1) % 2], output_buf[(i - 1) % 2], sycl_events[(i - 1) % 2], total_kernel_time_per_slot[(i - 1) % 2]);

    // Generate input for the next kernel.
    ProcessInput(input_buf[(i - 1) % 2]);
  }

  // Process output of the final kernel
  ProcessOutput(input_buf[(kTimes - 1) % 2], output_buf[(kTimes - 1) % 2], sycl_events[(kTimes - 1) % 2], total_kernel_time_per_slot[(kTimes - 1) % 2]);

  total_kernel_time = 0;
  for (int i = 0; i < 2; i++) {
    total_kernel_time += total_kernel_time_per_slot[i];
  }

  auto end = std::chrono::steady_clock::now();
  double time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

  std::cout << "\nOverall execution time with double buffering = "
            << (unsigned)(time_span * 1000)
            << " ms\n";
  std::cout << "Total kernel-only execution time with double buffering = "
            << (unsigned)(total_kernel_time / 1000000) << " ms\n";
  std::cout << "Throughput = " << std::setprecision(8)
            << (float)kSize * (float)kTimes * (float)sizeof(float) / (float)time_span / 1000000
            << " MB/s\n\n\n";
  
  if (pass) {
    std::cout << "Verification PASSED\n";
  } else {
    std::cout << "Verification FAILED\n";
    return 1;
  }
  return 0;
}

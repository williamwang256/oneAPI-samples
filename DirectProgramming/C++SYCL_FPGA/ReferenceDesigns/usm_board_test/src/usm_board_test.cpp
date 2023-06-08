//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <chrono>
#include <functional>
#include <iomanip>
#include <random>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

/**
 *  Testing the access bandwidth of a kernel launched with arguments that are
 *  host allocated.
 */

static const double MB = 1000.0 * 1000.0;
static const double GB = MB * 1000.0;

static const int TYPE_MEMCOPY = 0;
static const int TYPE_READ = 1;
static const int TYPE_WRITE = 2;

// ulong8 was picked because it is 64 bytes in size and that is the width of
// the interconnect to global memory
// using TYPE = sycl::ulong8;

class MemCopy;
class MemRead;
class MemWrite;

void memcopy_kernel(sycl::queue &q,
                    sycl::vec<long, 8> *in,
                    sycl::vec<long, 8> *out,
                    const sycl::range<1> numItems
) {
  q.single_task<class MemCopy>([=]() [[intel::kernel_args_restrict]] {
    sycl::host_ptr<sycl::vec<long, 8>> in_h(in);
    sycl::host_ptr<sycl::vec<long, 8>> out_h(out);
    for (size_t i = 0; i < numItems.get(0); i++) {
      out_h[i] = in_h[i];
    }
  });
}

void read_kernel(sycl::queue &q,
                 sycl::vec<long, 8> *in,
                 sycl::vec<long, 8> *out,
                 const sycl::range<1> numItems
) {
  q.single_task<class MemRead>([=]() {
    sycl::vec<long, 8> sum{0};
    sycl::host_ptr<sycl::vec<long, 8>> in_h(in);
    sycl::host_ptr<sycl::vec<long, 8>> out_h(out);
    for (size_t i = 0; i < numItems.get(0); i++) {
      sum += in_h[i];
    }
    // This prevents the reads from being optimized away
    out_h[0] = sum;
  });
}

void write_kernel(sycl::queue &q,
                  sycl::vec<long, 8> *in,
                  sycl::vec<long, 8> *out,
                  const sycl::range<1> numItems
) {
  q.single_task<class MemWrite>([=]() {
    sycl::vec<long, 8> anws{5};
    sycl::host_ptr<sycl::vec<long, 8>> out_h(out);
    for (size_t i = 0; i < numItems.get(0); i++) {
      out_h[i] = anws;
    }
  });
}

bool verify_memcopy_kernel(sycl::vec<long, 8> *in,
                           sycl::vec<long, 8> *out,
                           const sycl::range<1> numItems
) {
  for (auto i = 0; i < numItems.get(0); i++) {
    sycl::vec<long, 8> compare = in[i] == out[i];
    for (auto j = 0; j < compare.size(); j++) {
      if (!compare[j]) {
        std::cerr << "ERROR: Values do not match, in[" << i << "][" << j << "]:"
                  << in[i][j] << " != out[" << i << "][" << j << "]:"
                  << out[i][j] << std::endl;
        return false;
      }
    }
  }
  return true;
}

bool verify_read_kernel(sycl::vec<long, 8> *in,
                        sycl::vec<long, 8> *out,
                        const sycl::range<1> numItems
) {
  sycl::vec<long, 8> anwser{0};
  for (auto i = 0; i < numItems.get(0); i++) {
    anwser += in[i];
  }
  for (auto i = 0; i < numItems.get(0); i++) {
    sycl::vec<long, 8> compare{0};
    if (i == 0) {
      compare = anwser == out[i];
    } else {
      compare = compare == out[i];
    }
    for (auto j = 0; j < compare.size(); j++) {
      if (!compare[j]) {
        if (i == 0) {
          std::cerr << "ERROR: Values do not match, anwser[" << j << "]:" 
                    << anwser[j];
        } else {
          std::cerr << "ERROR: Values do not match, anwser[" << j << "]:"
                    << compare[j];
        }
        std::cerr << " != out[" << i << "][" << j << "]:" << out[i][j]
                  << std::endl;
        return false;
      }
    }
  }
  return true;
}

bool verify_write_kernel(sycl::vec<long, 8> *in,
                         sycl::vec<long, 8> *out,
                         const sycl::range<1> numItems
) {
  sycl::vec<long, 8> anwser{5};
  for (auto i = 0; i < numItems.get(0); i++) {
    sycl::vec<long, 8> compare = anwser == out[i];
    for (auto j = 0; j < compare.size(); j++) {
      if (!compare[j]) {
        std::cerr << "ERROR: Values do not match, anwser[" << j << "]:"
                  << anwser[j] << " != out[" << i << "][" << j << "]:"
                  << out[i][j] << std::endl;
        return false;
      }
    }
  }
  return true;
}

void run_test(
    sycl::queue &q,
    const size_t numBytes,
    int itterations,
    std::function<void(sycl::queue &, sycl::vec<long, 8> *, sycl::vec<long, 8> *, const sycl::range<1>)> kernel,
    std::function<bool(sycl::vec<long, 8> *, sycl::vec<long, 8> *, const sycl::range<1>)> verify,
    std::chrono::microseconds &time
) {

  const sycl::range<1> numItems{numBytes / sizeof(sycl::vec<long, 8>)};
  sycl::vec<long, 8> *in = sycl::malloc_host<sycl::vec<long, 8>>(numItems.get(0), q.get_context());
  sycl::vec<long, 8> *out = sycl::malloc_host<sycl::vec<long, 8>>(numItems.get(0), q.get_context());

  if (in == nullptr || out == nullptr) {
    std::cerr << "Error: Out of memory, can't allocate " << numBytes << " bytes"
              << std::endl;
    exit(1);
  }

  // initialize the input
  std::random_device rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<long> distrib(0, 1024);
  for (auto i = 0; i < numItems.get(0); i++) {
    in[i] = {distrib(gen)};
    out[i] = {0};
  }

  // The first invocation is slow because the following calls are called when
  // the kernel is submitted to the queue: clCreateBuffer,
  // clCreateProgramWithBinary, clBuildProgram, clEnqueueWriteBuffer(implicit),
  // clEnqueueNDsycl::rangeKernel.
  // Buffer creation, program creation, and device programming does not happen
  // on subsequent calls.
  kernel(q, in, out, numItems);
  q.wait();

  if (!verify(in, out, numItems)) {
    std::cerr << "FAILED" << std::endl;
    exit(1);
  }

  std::array<std::chrono::high_resolution_clock::time_point, 3> t;
  for (auto i = 0; i < itterations; i++) {
    t[0] = std::chrono::high_resolution_clock::now();
    kernel(q, in, out, numItems);
    q.wait();
    t[1] = std::chrono::high_resolution_clock::now();
    time += std::chrono::duration_cast<std::chrono::microseconds>(t[1] - t[0]);
  }

  sycl::free(in, q.get_context());
  sycl::free(out, q.get_context());
}

int main(int argc, char *argv[]) {

#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif
  sycl::queue q = sycl::queue(selector);

  sycl::platform platform = q.get_context().get_platform();
  sycl::device device = q.get_device();
  std::cout << "Platform name: "
            << platform.get_info<sycl::info::platform::name>().c_str()
            << std::endl;
  std::cout << "Device name: "
            << device.get_info<sycl::info::device::name>().c_str() << std::endl
            << std::endl
            << std::endl;

  int itterations = 1;
  size_t data_size = 1024 * 1024 * 1024;
  std::cout << "Iterations: " << itterations << std::endl;
  std::cout << "Data size: " << data_size / MB << " MB" << std::endl;
  std::cout << "Data type size: " << sizeof(sycl::vec<long, 8>) << " bytes" << std::endl;

  std::cout << "-- Results Full Duplex -- " << std::endl;
  for (int i = 0; i < 3; i++) {
    std::chrono::microseconds time{0};
    switch (i) {
    case TYPE_MEMCOPY: {
      std::cout << std::endl << "Case: Full Duplex" << std::endl;
      std::function<void(sycl::queue &, sycl::vec<long, 8> *, sycl::vec<long, 8> *, const sycl::range<1>)> memcopy_k = memcopy_kernel;
      std::function<bool(sycl::vec<long, 8> *, sycl::vec<long, 8> *, const sycl::range<1>)> verify = verify_memcopy_kernel;
      run_test(q, data_size, itterations, memcopy_k, verify, time);
    } break;
    case TYPE_READ: {
      std::cout << std::endl << "Case: From Host to Device" << std::endl;
      std::function<void(sycl::queue &, sycl::vec<long, 8> *, sycl::vec<long, 8> *, const sycl::range<1>)> read_k = read_kernel;
      std::function<bool(sycl::vec<long, 8> *, sycl::vec<long, 8> *, const sycl::range<1>)> verify = verify_read_kernel;
      run_test(q, data_size, itterations, read_k, verify, time);
    } break;
    case TYPE_WRITE: {
      std::cout << std::endl << "Case: From Device to Host" << std::endl;
      std::function<void(sycl::queue &, sycl::vec<long, 8> *, sycl::vec<long, 8> *, const sycl::range<1>)> write_k = write_kernel;
      std::function<bool(sycl::vec<long, 8> *, sycl::vec<long, 8> *, const sycl::range<1>)> verify = verify_write_kernel;
      run_test(q, data_size, itterations, write_k, verify, time);
    } break;
    default:
      std::cout << "Error: Don't know how to launch test " << i << std::endl;
      return 1;
    }

    time /= itterations;
    std::cout << "Average Time: " << time.count() / 1000.0 << " ms\t"
              << std::endl;

    double data_size_gb;
    if (i == TYPE_MEMCOPY) {
      // full duplex transfers twice the amount of data
      data_size_gb = data_size * 2 / GB;
    } else {
      data_size_gb = data_size / GB;
    }
    std::cout << "Average Throughput: "
              << (data_size_gb / (time.count() / (1000.0 * 1000.0)))
              << " GB/s\t" << std::endl;

    // For QOR set to parse
    if (i == TYPE_MEMCOPY) {
      std::cout << "Throughput: "
                << (data_size_gb / (time.count() / (1000.0 * 1000.0)))
                << " GB/s\t" << std::endl;
    }
  }

  return 0;
}
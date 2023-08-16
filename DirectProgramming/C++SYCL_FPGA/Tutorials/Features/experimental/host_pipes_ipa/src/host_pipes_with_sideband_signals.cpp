#include <iomanip>
#include <iostream>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/prototype/pipes_ext.hpp>
#include <sycl/sycl.hpp>

// Forward declare the kernel and pipe names
// (This prevents unwanted name mangling in the optimization report.)
class InputPipe;
class OutputPipe;
class KernelCompute;

// Arbitrary computation function
int SomethingComplicated(int val) { return (int)(val * sqrt(val)); }

constexpr int kBitsPerSymbol = 8;
constexpr int kSymbolsPerBeat = 8;
using PipeDataT = ac_int<kBitsPerSymbol * kSymbolsPerBeat, false>;
using StreamingBeatT =
    sycl::ext::intel::experimental::StreamingBeat<PipeDataT, true, true>;

// Host pipe properties
using PipePropertiesT = decltype(sycl::ext::oneapi::experimental::properties(
    sycl::ext::intel::experimental::ready_latency<0>,
    sycl::ext::intel::experimental::bits_per_symbol<kBitsPerSymbol>,
    sycl::ext::intel::experimental::uses_valid<true>,
    sycl::ext::intel::experimental::first_symbol_in_high_order_bits<true>,
    sycl::ext::intel::experimental::protocol_avalon_streaming_uses_ready));

using PipeIn = sycl::ext::intel::experimental::pipe<
    InputPipe,      // An identifier for the pipe
    StreamingBeatT, // The type of data in the pipe
    8,              // The capacity of the pipe
    PipePropertiesT // Customizable pipe properties
    >;

using PipeOut = sycl::ext::intel::experimental::pipe<
    OutputPipe,     // An identifier for the pipe
    StreamingBeatT, // The type of data in the pipe
    8,              // The capacity of the pipe
    PipePropertiesT // Customizable pipe properties
    >;

// A kernel with streaming input and output interfaces. Reads a data element
// from PipeIn, processes it, and writes the result to PipeOut.
struct Kernel {
  int count;

  void operator()() const {
    for (size_t i = 0; i < count; i++) {

      StreamingBeatT b_in = PipeIn::read();
      PipeDataT d = b_in.data;
      bool sop = b_in.sop;
      bool eop = b_in.eop;
      int empty = b_in.empty;

      auto r = SomethingComplicated(d);

      StreamingBeatT b_out(r, sop, eop, empty);
      PipeOut::write(b_out);
    }
  }
};

int main() {

  try {

#if FPGA_SIMULATOR
    auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
    auto selector = sycl::ext::intel::fpga_selector_v;
#else // #if FPGA_EMULATOR
    auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif
    sycl::queue q(selector);
    auto device = q.get_device();
    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;

    int count = 16;

    // Generate input data for the kernel
    for (int i = 0; i < count; i++) {
      StreamingBeatT b_in(i, i == 0, i == count - 1, 0);
      PipeIn::write(q, b_in);
    }

    q.single_task<KernelCompute>(Kernel{count});

    // Verify functional correctness
    // (no need to wait on kernel to finish as the pipe reads are blocking)
    bool passed = true;
    for (int i = 0; i < count; i++) {
      StreamingBeatT b_out = PipeOut::read(q);
      PipeDataT val_device = b_out.data;
      bool sop = b_out.sop;
      bool eop = b_out.eop;
      int empty = b_out.empty;
      std::cout << "Data: "  << std::setw(2) << (int)(val_device)
                << "; sop: " << std::setw(2) << sop
                << "; eop: " << std::setw(2) << eop
                << "; empty: " << empty << std::endl;
      int val_host = SomethingComplicated(i);
      passed &= (val_device == val_host);
      if (val_device != val_host) {
        std::cout << "Error: expected " << val_host << ", got "
                  << (int)(val_device) << std::endl;
      }
    }
    std::cout << std::endl << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed;

  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";
    std::terminate();
  }
}
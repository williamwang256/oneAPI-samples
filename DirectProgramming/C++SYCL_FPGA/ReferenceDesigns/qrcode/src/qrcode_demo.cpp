#include <stdio.h>

#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#include "identify.hpp"
#include "image_processing.hpp"
#include "quirc.hpp"
#include "decode_fpga.hpp"

#ifdef __SYCL_DEVICE_ONLY__
  #define CL_CONSTANT __attribute__((opencl_constant))
#else
  #define CL_CONSTANT
#endif
#define PRINTF(format, ...) { \
      static const CL_CONSTANT char _format[] = format; \
      sycl::ext::oneapi::experimental::printf(_format, ## __VA_ARGS__); }


// class IdentifierKernel {
// public:
//   uint8_t *image_ptr;

//   void operator()() const {
//     constexpr int kW = 300;
//     constexpr int kH = 300;

//     [[intel::fpga_memory]] uint8_t image_mem[kW * kH];
//     [[intel::fpga_memory]] uint8_t pixels_mem[kW * kH];
//     for (int i = 0; i < kW * kH; i++) {
//       image_mem[i] = image_ptr[i];
//       pixels_mem[i] = 0;
//     }

//     constexpr size_t kNumFloodFillVars = ((kH * 2 / 3) == 0) ? 1 : (kH * 2 / 3);
//     constexpr size_t kVarsByteSize = sizeof(quirc_flood_fill_vars) * kNumFloodFillVars;
//     struct quirc_flood_fill_vars floodFillVarsMem[kVarsByteSize];
//     struct quirc_region regions[QUIRC_MAX_REGIONS];

//     int numRegions = QUIRC_PIXEL_REGION;
//     constexpr int kNumCapstones = 0;
//     constexpr int kNumGrids = 0;

//     uint8_t threshold = otsu<kW, kH>(&image_mem[0]);
//     pixels_setup<kW, kH>(&image_mem[0], &pixels_mem[0], threshold);

//     for (int i = 0; i < kH; i++) {
//       finder_scan<kW, kH>(&pixels_mem[0], &numRegions, &regions[0], i);
//     }
//   }
// };

template <int size>
class DecoderKernel {
public:
  struct quirc_point corner1;
  struct quirc_point corner2;
  struct quirc_point corner3;
  struct quirc_point corner4;
  uint8_t *cell_bitmap;

  void operator()() const {
    quirc_decode_error_t err;
    struct datastream ds;

    int ecc_level;
    int mask;
    int data_type;
    uint8_t payload[QUIRC_MAX_PAYLOAD];
    int payload_len;
    uint32_t eci;

    constexpr int version = (size - 17) / 4;

    uint8_t cell_bitmap_local[QUIRC_MAX_BITMAP];
    for (int i = 0; i < QUIRC_MAX_BITMAP; i++) {
      cell_bitmap_local[i] = cell_bitmap[i];
    }

    // Read format information -- try both locations
    err = read_format(size, &cell_bitmap_local[0], ecc_level, mask, 0);
    if (err) {
      err = read_format(size, &cell_bitmap_local[0], ecc_level, mask, 1);
    }

    ds.raw = &payload[0];

    read_data<version, size>(&cell_bitmap_local[0], mask, &ds);
    err = codestream_ecc<version>(ecc_level, &ds);
    ds.raw = NULL;
    err = decode_payload<version>(data_type, &payload[0], payload_len, eci, &ds);

    PRINTF("Data: %s\n", payload);
  }
};



int main(int argc, char *argv[]) {

#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif
  sycl::queue q = sycl::queue(selector);  

  struct quirc *decoder = (struct quirc *) malloc(sizeof(*decoder));
  memset(decoder, 0, sizeof(*decoder));
  load_png(decoder, "/p/psg/swip/w/wangwil3/qrcode/image.png");
  quirc_end(decoder);

  // q.single_task<class Identifier>(IdentifierKernel{decoder->image});

  struct quirc_code code;

  quirc_extract(decoder, 0, &code);

  // struct quirc_data data;
  // quirc_decode(&code, &data);

  q.single_task<class Decoder>(DecoderKernel<33>{code.corners[0], code.corners[1], code.corners[2], code.corners[3], code.cell_bitmap});

  quirc_destroy(decoder);
  return 0;
}
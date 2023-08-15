#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

#ifdef __SYCL_DEVICE_ONLY__
  #define CL_CONSTANT __attribute__((opencl_constant))
#else
  #define CL_CONSTANT
#endif
#define PRINTF(format, ...) { \
      static const CL_CONSTANT char _format[] = format; \
      sycl::ext::oneapi::experimental::printf(_format, ## __VA_ARGS__); }

template<int w, int h>
uint8_t otsu(uint8_t *image_ptr) {
  constexpr unsigned int numPixels = w * h;

  // Calculate histogram
  [[intel::fpga_memory]] unsigned int histogram[UINT8_MAX + 1];
  (void) memset(histogram, 0, sizeof(histogram));
  for (int i = 0; i < numPixels; i++) {
    histogram[image_ptr[i]]++;
  }

  // Calculate weighted sum of histogram values
  float sum = 0;
#pragma unroll
  for (int i = 0; i <= UINT8_MAX; ++i) {
    sum += i * histogram[i];
  }

  // Compute threshold
  float sumB[3] = {0, 0, 0};
  unsigned int q1 = 0;
  float max = 0;
  uint8_t threshold = 0;

  [[intel::initiation_interval(1)]]
  [[intel::ivdep]]
  for (int i = 0; i <= UINT8_MAX; ++i) {
    // Weighted background
    q1 += histogram[i];
    if (q1 == 0)
      continue;

    // Weighted foreground
    const unsigned int q2 = numPixels - q1;
    if (q2 == 0)
      break;

    float tmp = sumB[0];
    tmp += i * histogram[i];
#pragma unroll
    for (int i = 0; i < 2; i++) {
      sumB[i] = sumB[i + 1];
    }
    sumB[2] = tmp;
    
    float runningSum = 0;
#pragma unroll
    for (int i = 0; i < 3; i++) {
      runningSum += sumB[i];
    }

    const float m1 = runningSum / q1;
    const float m2 = (sum - runningSum) / q2;
    const float m1m2 = m1 - m2;
    const float variance = m1m2 * m1m2 * q1 * q2;
    if (variance >= max) {
      threshold = i;
      max = variance;
    }
  }

  return threshold;
}

template<int w, int h>
void pixels_setup(uint8_t *image_ptr, uint8_t *pixels_ptr, uint8_t threshold) {
  for (int i = 0; i < w * h; i++) {
    pixels_ptr[i] = (image_ptr[i] < threshold) ? QUIRC_PIXEL_BLACK : QUIRC_PIXEL_WHITE;
  }
}

// typedef void (*span_func_t)(void *user_data, int y, int left, int right);

// template<void (*func)(void *user_data, int y, int left, int right)>
// void flood_fill_seed(struct quirc *q, int x0, int y0, int from, int to, void *user_data) {
//   struct quirc_flood_fill_vars *const stack = q->flood_fill_vars;
//   const size_t stack_size = q->num_flood_fill_vars;
//   const struct quirc_flood_fill_vars *const last_vars = &stack[stack_size - 1];

//   QUIRC_ASSERT(from != to);
//   QUIRC_ASSERT(q->pixels[y0 * q->w + x0] == from);

//   struct quirc_flood_fill_vars *next_vars;
//   int next_left;

//   /* Set up the first context  */
//   next_vars = stack;
//   next_vars->y = y0;

//   /* Fill the extent */
//   flood_fill_line(q, x0, next_vars->y, from, to, func, user_data, &next_left,
//                   &next_vars->right);
//   next_vars->left_down = next_left;
//   next_vars->left_up = next_left;

//   while (true) {
//     struct quirc_flood_fill_vars *const vars = next_vars;
//     uint8_t *row;

//     if (vars == last_vars) {
//       /*
//        * "Stack overflow".
//        * Just stop and return.
//        * This can be caused by very complex shapes in
//        * the image, which is not likely a part of
//        * a valid QR code anyway.
//        */
//       break;
//     }

//     /* Seed new flood-fills */
//     if (vars->y > 0) {
//       row = q->pixels + (vars->y - 1) * q->w;

//       next_vars =
//           flood_fill_call_next(q, row, from, to, func, user_data, vars, -1);
//       if (next_vars != NULL) {
//         continue;
//       }
//     }

//     if (vars->y < q->h - 1) {
//       row = q->pixels + (vars->y + 1) * q->w;

//       next_vars =
//           flood_fill_call_next(q, row, from, to, func, user_data, vars, 1);
//       if (next_vars != NULL) {
//         continue;
//       }
//     }

//     if (vars > stack) {
//       /* Restore the previous context */
//       next_vars = vars - 1;
//       continue;
//     }

//     /* We've done. */
//     break;
//   }
// }

template<int w, int h>
int region_code(uint8_t *pixels_ptr, int *num_regions, struct quirc_region *regions, int x, int y) {
  int pixel;
  struct quirc_region *box;
  int region;

  if (x < 0 || y < 0 || x >= w || y >= h)
    return -1;

  pixel = pixels_ptr[y * w + x];

  if (pixel >= QUIRC_PIXEL_REGION)
    return pixel;

  if (pixel == QUIRC_PIXEL_WHITE)
    return -1;

  if (*num_regions >= QUIRC_MAX_REGIONS)
    return -1;

  region = *num_regions;
  box = &regions[(*num_regions)++];

  memset(box, 0, sizeof(*box));

  box->seed.x = x;
  box->seed.y = y;
  box->capstone = -1;

  // flood_fill_seed<area_count>(x, y, pixel, region, box);

  return region;
}

template<int w, int h>
void test_capstone(uint8_t *pixels_ptr, int *num_regions, struct quirc_region *regions, unsigned int x, unsigned int y, unsigned int *pb) {
  int ring_right = region_code<w, h>(pixels_ptr, num_regions, regions, x - pb[4], y);
  int stone      = region_code<w, h>(pixels_ptr, num_regions, regions, x - pb[4] - pb[3] - pb[2], y);
  int ring_left  = region_code<w, h>(pixels_ptr, num_regions, regions, x - pb[4] - pb[3] - pb[2] - pb[1] - pb[0], y);
  
  // struct quirc_region *stone_reg;
  // struct quirc_region *ring_reg;
  // unsigned int ratio;

  // if (ring_left < 0 || ring_right < 0 || stone < 0)
  //   return;

  // /* Left and ring of ring should be connected */
  // if (ring_left != ring_right)
  //   return;

  // /* Ring should be disconnected from stone */
  // if (ring_left == stone)
  //   return;

  // stone_reg = &regions[stone];
  // ring_reg = &regions[ring_left];

  // /* Already detected */
  // if (stone_reg->capstone >= 0 || ring_reg->capstone >= 0)
  //   return;

  // /* Ratio should ideally be 37.5 */
  // ratio = stone_reg->count * 100 / ring_reg->count;
  // if (ratio < 10 || ratio > 70)
  //   return;

  // record_capstone(ring_left, stone);
}

template<int w, int h>
void finder_scan(uint8_t *pixels_ptr, int *num_regions, struct quirc_region *regions, unsigned int y) {
  uint8_t *row = pixels_ptr + y * w;
  unsigned int x;
  int last_color = 0;
  unsigned int run_length = 0;
  unsigned int run_count = 0;
  [[intel::fpga_register]] unsigned int pb[5];

  memset(pb, 0, sizeof(pb));
  for (x = 0; x < w; x++) {
    int color = row[x] ? 1 : 0;

    if (x && color != last_color) {
      memmove(pb, pb + 1, sizeof(pb[0]) * 4); // shiftreg?
      pb[4] = run_length;
      run_length = 0;
      run_count++;

      if (!color && run_count >= 5) {
        const int scale = 16;
        static const unsigned int check[5] = {1, 1, 3, 1, 1};
        unsigned int avg, err;
        unsigned int i;
        int ok = 1;

        avg = (pb[0] + pb[1] + pb[3] + pb[4]) * scale / 4;
        err = avg * 3 / 4;

        for (i = 0; i < 5; i++) {
          if (pb[i] * scale < check[i] * avg - err || pb[i] * scale > check[i] * avg + err) {
            ok = 0;
          }
        }

        if (ok) {
          test_capstone<w, h>(pixels_ptr, num_regions, regions, x, y, pb);
        }
      }
    }

    run_length++;
    last_color = color;
  }
}
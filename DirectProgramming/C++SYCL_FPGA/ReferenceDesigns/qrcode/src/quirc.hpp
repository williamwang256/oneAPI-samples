/* quirc -- QR-code recognition library
 * Copyright (C) 2010-2012 Daniel Beer <dlbeer@gmail.com>
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
 * WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
 * ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
 * OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#ifndef __QUIRC_HPP__
#define __QUIRC_HPP__

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

struct quirc;

/* This structure describes a location in the input image buffer. */
struct quirc_point {
  int x;
  int y;
};

/* This enum describes the various decoder errors which may occur. */
typedef enum {
  QUIRC_SUCCESS = 0,
  QUIRC_ERROR_INVALID_GRID_SIZE,
  QUIRC_ERROR_INVALID_VERSION,
  QUIRC_ERROR_FORMAT_ECC,
  QUIRC_ERROR_DATA_ECC,
  QUIRC_ERROR_UNKNOWN_DATA_TYPE,
  QUIRC_ERROR_DATA_OVERFLOW,
  QUIRC_ERROR_DATA_UNDERFLOW
} quirc_decode_error_t;

/* Return a string error message for an error code. */
const char *quirc_strerror(quirc_decode_error_t err);

/* Limits on the maximum size of QR-codes and their content. */
#define QUIRC_MAX_VERSION 40
#define QUIRC_MAX_GRID_SIZE (QUIRC_MAX_VERSION * 4 + 17)
#define QUIRC_MAX_BITMAP (((QUIRC_MAX_GRID_SIZE * QUIRC_MAX_GRID_SIZE) + 7) / 8)
#define QUIRC_MAX_PAYLOAD 8896

/* QR-code ECC types. */
#define QUIRC_ECC_LEVEL_M 0
#define QUIRC_ECC_LEVEL_L 1
#define QUIRC_ECC_LEVEL_H 2
#define QUIRC_ECC_LEVEL_Q 3

/* QR-code data types. */
#define QUIRC_DATA_TYPE_NUMERIC 1
#define QUIRC_DATA_TYPE_ALPHA 2
#define QUIRC_DATA_TYPE_BYTE 4
#define QUIRC_DATA_TYPE_KANJI 8

/* Common character encodings */
#define QUIRC_ECI_ISO_8859_1 1
#define QUIRC_ECI_IBM437 2
#define QUIRC_ECI_ISO_8859_2 4
#define QUIRC_ECI_ISO_8859_3 5
#define QUIRC_ECI_ISO_8859_4 6
#define QUIRC_ECI_ISO_8859_5 7
#define QUIRC_ECI_ISO_8859_6 8
#define QUIRC_ECI_ISO_8859_7 9
#define QUIRC_ECI_ISO_8859_8 10
#define QUIRC_ECI_ISO_8859_9 11
#define QUIRC_ECI_WINDOWS_874 13
#define QUIRC_ECI_ISO_8859_13 15
#define QUIRC_ECI_ISO_8859_15 17
#define QUIRC_ECI_SHIFT_JIS 20
#define QUIRC_ECI_UTF_8 26

/* This structure is used to return information about detected QR codes
 * in the input image.
 */
struct quirc_code {
  /* The four corners of the QR-code, from top left, clockwise */
  struct quirc_point corners[4];

  /* The number of cells across in the QR-code. The cell bitmap
   * is a bitmask giving the actual values of cells. If the cell
   * at (x, y) is black, then the following bit is set:
   *
   *     cell_bitmap[i >> 3] & (1 << (i & 7))
   *
   * where i = (y * size) + x.
   */
  int size;
  uint8_t cell_bitmap[QUIRC_MAX_BITMAP];
};

/* This structure holds the decoded QR-code data */
struct quirc_data {
  /* Various parameters of the QR-code. These can mostly be
   * ignored if you only care about the data.
   */
  int version;
  int ecc_level;
  int mask;

  /* This field is the highest-valued data type found in the QR
   * code.
   */
  int data_type;

  /* Data payload. For the Kanji datatype, payload is encoded as
   * Shift-JIS. For all other datatypes, payload is ASCII text.
   */
  uint8_t payload[QUIRC_MAX_PAYLOAD];
  int payload_len;

  /* ECI assignment number */
  uint32_t eci;
};

#define QUIRC_ASSERT(a) assert(a)

#define QUIRC_PIXEL_WHITE 0
#define QUIRC_PIXEL_BLACK 1
#define QUIRC_PIXEL_REGION 2

#ifndef QUIRC_MAX_REGIONS
#define QUIRC_MAX_REGIONS 254
#endif
#define QUIRC_MAX_CAPSTONES 32
#define QUIRC_MAX_GRIDS (QUIRC_MAX_CAPSTONES * 2)

#define QUIRC_PERSPECTIVE_PARAMS 8

#if QUIRC_MAX_REGIONS < UINT8_MAX
#define QUIRC_PIXEL_ALIAS_IMAGE 1
typedef uint8_t quirc_pixel_t;
#elif QUIRC_MAX_REGIONS < UINT16_MAX
#define QUIRC_PIXEL_ALIAS_IMAGE 0
typedef uint16_t quirc_pixel_t;
#else
#error "QUIRC_MAX_REGIONS > 65534 is not supported"
#endif

struct quirc_region {
  struct quirc_point seed;
  int count;
  int capstone;
};

struct quirc_capstone {
  int ring;
  int stone;

  struct quirc_point corners[4];
  struct quirc_point center;
  float c[QUIRC_PERSPECTIVE_PARAMS];

  int qr_grid;
};

struct quirc_grid {
  /* Capstone indices */
  int caps[3];

  /* Alignment pattern region and corner */
  int align_region;
  struct quirc_point align;

  /* Timing pattern endpoints */
  struct quirc_point tpep[3];

  /* Grid size and perspective transform */
  int grid_size;
  float c[QUIRC_PERSPECTIVE_PARAMS];
};

struct quirc_flood_fill_vars {
  int y;
  int right;
  int left_up;
  int left_down;
};

struct quirc {
  uint8_t *image;
  quirc_pixel_t *pixels;
  int w;
  int h;

  int num_regions;
  struct quirc_region regions[QUIRC_MAX_REGIONS];

  int num_capstones;
  struct quirc_capstone capstones[QUIRC_MAX_CAPSTONES];

  int num_grids;
  struct quirc_grid grids[QUIRC_MAX_GRIDS];

  size_t num_flood_fill_vars;
  struct quirc_flood_fill_vars *flood_fill_vars;
};

/************************************************************************
 * QR-code version information database
 */

#define QUIRC_MAX_VERSION 40
#define QUIRC_MAX_ALIGNMENT 7

struct quirc_rs_params {
  int bs; /* Small block size */
  int dw; /* Small data words */
  int ns; /* Number of small blocks */
};

struct quirc_version_info {
  int data_bytes;
  int apat[QUIRC_MAX_ALIGNMENT];
  struct quirc_rs_params ecc[4];
};

// extern constexpr struct quirc_version_info quirc_version_db[QUIRC_MAX_VERSION + 1];

#endif /* __QUIRC_HPP__ */
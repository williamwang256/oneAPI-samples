#ifndef __IMAGE_PROCESSING_HPP__
#define __IMAGE_PROCESSING_HPP__

#include <jpeglib.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "quirc.hpp"

/**
 * These functions are used to process images for QR-code recognition.
 * quirc_begin() must first be called to obtain access to a buffer into which
 * the input image should be placed. Optionally, the current width and height
 * may be returned.
 *
 * After filling the buffer, quirc_end() should be called to process
 * the image for QR-code recognition. The locations and content of each
 * code may be obtained using accessor functions described below.
 * 
 */
uint8_t *quirc_begin(struct quirc *q, int *w, int *h) {
  q->num_regions = QUIRC_PIXEL_REGION;
  q->num_capstones = 0;
  q->num_grids = 0;

  if (w)
    *w = q->w;
  if (h)
    *h = q->h;

  return q->image;
}

/**
 * Construct a new QR-code recognizer. This function will return NULL if
 * sufficient memory could not be allocated.
 *
 */
struct quirc *quirc_new(void) {
  struct quirc *q = (struct quirc *)malloc(sizeof(*q));

  if (!q)
    return NULL;

  memset(q, 0, sizeof(*q));
  return q;
}

/**
 * Destroy a QR-code recognizer.
 * 
 */
void quirc_destroy(struct quirc *q) {
  free(q->image);
  /* q->pixels may alias q->image when their type representation is of the
     same size, so we need to be careful here to avoid a double free */
  if (!QUIRC_PIXEL_ALIAS_IMAGE)
    free(q->pixels);
  free(q->flood_fill_vars);
  free(q);
}

/**
 * Resize the QR-code recognizer. The size of an image must be specified before
 * codes can be analyzed.
 *
 * This function returns 0 on success, or -1 if sufficient memory could not be
 * allocated.
 * 
 */
int quirc_resize(struct quirc *q, int w, int h) {
  uint8_t *image = NULL;
  quirc_pixel_t *pixels = NULL;
  size_t num_vars;
  size_t vars_byte_size;
  struct quirc_flood_fill_vars *vars = NULL;

  /*
   * XXX: w and h should be size_t (or at least unsigned) as negatives
   * values would not make much sense. The downside is that it would break
   * both the API and ABI. Thus, at the moment, let's just do a sanity
   * check.
   */
  if (w < 0 || h < 0) {
    free(image);
    free(pixels);
    free(vars);
    return -1;
  }

  /*
   * alloc a new buffer for q->image. We avoid realloc(3) because we want
   * on failure to be leave `q` in a consistant, unmodified state.
   */
  image = (uint8_t *)calloc(w, h);
  if (!image) {
    free(image);
    free(pixels);
    free(vars);
    return -1;
  }

  /* compute the "old" (i.e. currently allocated) and the "new"
     (i.e. requested) image dimensions */
  size_t olddim = q->w * q->h;
  size_t newdim = w * h;
  size_t min = (olddim < newdim ? olddim : newdim);

  /*
   * copy the data into the new buffer, avoiding (a) to read beyond the
   * old buffer when the new size is greater and (b) to write beyond the
   * new buffer when the new size is smaller, hence the min computation.
   */
  (void)memcpy(image, q->image, min);

  /* alloc a new buffer for q->pixels if needed */
  if (!QUIRC_PIXEL_ALIAS_IMAGE) {
    pixels = (quirc_pixel_t *)calloc(newdim, sizeof(quirc_pixel_t));
    if (!pixels) {
      free(image);
      free(pixels);
      free(vars);
      return -1;
    }
  }

  /*
   * alloc the work area for the flood filling logic.
   *
   * the size was chosen with the following assumptions and observations:
   *
   * - rings are the regions which requires the biggest work area.
   * - they consumes the most when they are rotated by about 45 degree.
   *   in that case, the necessary depth is about (2 * height_of_the_ring).
   * - the maximum height of rings would be about 1/3 of the image height.
   */

  if ((size_t)h * 2 / 2 != h) {
    free(image);
    free(pixels);
    free(vars);
    return -1; /* size_t overflow */
  }
  num_vars = (size_t)h * 2 / 3;
  if (num_vars == 0) {
    num_vars = 1;
  }

  vars_byte_size = sizeof(*vars) * num_vars;
  if (vars_byte_size / sizeof(*vars) != num_vars) {
    free(image);
    free(pixels);
    free(vars);
    return -1; /* size_t overflow */
  }
  vars = (quirc_flood_fill_vars *)malloc(vars_byte_size);
  if (!vars) {
    free(image);
    free(pixels);
    free(vars);
    return -1;
  }

  /* alloc succeeded, update `q` with the new size and buffers */
  q->w = w;
  q->h = h;
  free(q->image);
  q->image = image;
  if (!QUIRC_PIXEL_ALIAS_IMAGE) {
    free(q->pixels);
    q->pixels = pixels;
  }
  free(q->flood_fill_vars);
  q->flood_fill_vars = vars;
  q->num_flood_fill_vars = num_vars;

  return 0;
}

int load_png(struct quirc *q, const char *filename) {

  int width, height, rowbytes, interlace_type, number_passes = 1;
  png_uint_32 trns;
  png_byte color_type, bit_depth;
  png_structp png_ptr = NULL;
  png_infop info_ptr = NULL;
  FILE *infile = NULL;
  uint8_t *image;
  int ret = -1;
  int pass;

  if ((infile = fopen(filename, "rb")) == NULL) {
    if (png_ptr) {
      if (info_ptr)
        png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
      else
        png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
    }
    if (infile)
      fclose(infile);
    return (ret);
  }

  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (!png_ptr) {
    if (png_ptr) {
      if (info_ptr)
        png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
      else
        png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
    }
    if (infile)
      fclose(infile);
    return (ret);
  }

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    if (png_ptr) {
      if (info_ptr)
        png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
      else
        png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
    }
    if (infile)
      fclose(infile);
    return (ret);
  }

  if (setjmp(png_jmpbuf(png_ptr))) {
    if (png_ptr) {
      if (info_ptr)
        png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
      else
        png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
    }
    if (infile)
      fclose(infile);
    return (ret);
  }

  png_init_io(png_ptr, infile);

  png_read_info(png_ptr, info_ptr);

  color_type = png_get_color_type(png_ptr, info_ptr);
  bit_depth = png_get_bit_depth(png_ptr, info_ptr);
  interlace_type = png_get_interlace_type(png_ptr, info_ptr);

  // Read any color_type into 8bit depth, Grayscale format.
  // See http://www.libpng.org/pub/png/libpng-manual.txt

  // PNG_COLOR_TYPE_GRAY_ALPHA is always 8 or 16bit depth.
  if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
    png_set_expand_gray_1_2_4_to_8(png_ptr);

  if ((trns = png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)))
    png_set_tRNS_to_alpha(png_ptr);

  if (bit_depth == 16)
#if PNG_LIBPNG_VER >= 10504
    png_set_scale_16(png_ptr);
#else
    png_set_strip_16(png_ptr);
#endif

  if ((trns) || color_type & PNG_COLOR_MASK_ALPHA)
    png_set_strip_alpha(png_ptr);

  if (color_type == PNG_COLOR_TYPE_PALETTE)
    png_set_palette_to_rgb(png_ptr);

  if (color_type == PNG_COLOR_TYPE_PALETTE ||
      color_type == PNG_COLOR_TYPE_RGB ||
      color_type == PNG_COLOR_TYPE_RGB_ALPHA) {
    png_set_rgb_to_gray_fixed(png_ptr, 1, -1, -1);
  }

  if (interlace_type != PNG_INTERLACE_NONE)
    number_passes = png_set_interlace_handling(png_ptr);

  png_read_update_info(png_ptr, info_ptr);

  width = png_get_image_width(png_ptr, info_ptr);
  height = png_get_image_height(png_ptr, info_ptr);
  rowbytes = png_get_rowbytes(png_ptr, info_ptr);
  if (rowbytes != width) {
    fprintf(stderr, "load_png: expected rowbytes to be %u but got %u\n", width,
            rowbytes);
    if (png_ptr) {
      if (info_ptr)
        png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
      else
        png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
    }
    if (infile)
      fclose(infile);
    return (ret);
  }

  if (quirc_resize(q, width, height) < 0) {
    if (png_ptr) {
      if (info_ptr)
        png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
      else
        png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
    }
    if (infile)
      fclose(infile);
    return (ret);
  }


  q->num_regions = QUIRC_PIXEL_REGION;
  q->num_capstones = 0;
  q->num_grids = 0;
  image = q->image;

  for (pass = 0; pass < number_passes; pass++) {
    int y;

    for (y = 0; y < height; y++) {
      png_bytep row_pointer = image + y * width;
      png_read_rows(png_ptr, &row_pointer, NULL, 1);
    }
  }

  png_read_end(png_ptr, info_ptr);

  if (png_ptr) {
    if (info_ptr)
      png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
    else
      png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
  }
  if (infile)
    fclose(infile);
  return (ret);
}

#endif /* __IMAGE_PROCESSING_HPP__ */
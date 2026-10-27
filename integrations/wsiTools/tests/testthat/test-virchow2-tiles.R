testthat::test_that("Virchow2 tiling respects anisotropic microscopy spacing", {
  slide <- wsiTools:::wsi_mock_slide(width = 1000, height = 600, levels = 1)
  slide$properties[["openslide.mpp-x"]] <- "0.25"
  slide$properties[["openslide.mpp-y"]] <- "0.5"
  geometry <- wsi_virchow2_geometry(slide)
  testthat::expect_equal(geometry$native_tile_wh, c(448L, 224L))
  testthat::expect_equal(geometry$effective_mpp_xy, c(0.5, 0.5))
  grid <- wsi_virchow2_tile_plan(slide)
  testthat::expect_equal(nrow(grid), 9L)
  testthat::expect_equal(grid$tile_index, 0:8)
  testthat::expect_equal(max(grid$x + grid$valid_width), 1000L)
  testthat::expect_equal(max(grid$y + grid$valid_height), 600L)
  testthat::expect_true(all(grid$pad_right >= 0L & grid$pad_bottom >= 0L))
  testthat::expect_equal(grid$x[grid$row == 0L], c(0L, 448L, 896L))
})

testthat::test_that("Virchow2 tile reader returns calibrated RGB and white edge padding", {
  testthat::skip_if_not_installed("magick")
  slide <- wsiTools:::wsi_mock_slide(width = 550, height = 520, levels = 1)
  slide$properties[["openslide.mpp-x"]] <- "0.25"
  slide$properties[["openslide.mpp-y"]] <- "0.25"
  plan <- wsi_virchow2_tile_plan(slide)
  tile <- wsi_virchow2_tile(slide, plan[nrow(plan), , drop = FALSE])
  testthat::expect_equal(dim(tile), c(224L, 224L, 3L))
  testthat::expect_type(tile, "integer")
  testthat::expect_true(all(tile >= 0L & tile <= 255L))
  testthat::expect_true(any(tile == 255L))
  testthat::expect_error(wsi_virchow2_tile(slide, plan), "exactly one")
})

testthat::test_that("Spot-centered tile plan pads all image edges", {
  testthat::skip_if_not_installed("magick")
  slide <- wsiTools:::wsi_mock_slide(width = 550, height = 520, levels = 1)
  slide$properties[["openslide.mpp-x"]] <- "0.25"
  slide$properties[["openslide.mpp-y"]] <- "0.25"
  centers <- data.frame(x = c(0, 549, 275), y = c(0, 519, 260),
                        spot_id = c("first", "last", "middle"))
  plan <- wsi_virchow2_spot_tile_plan(slide, centers)
  testthat::expect_equal(nrow(plan), 3L)
  testthat::expect_true(plan$pad_left[[1L]] > 0L && plan$pad_top[[1L]] > 0L)
  testthat::expect_true(plan$pad_right[[2L]] > 0L && plan$pad_bottom[[2L]] > 0L)
  testthat::expect_equal(plan$spot_id, centers$spot_id)
  for (i in seq_len(nrow(plan))) {
    tile <- wsi_virchow2_tile(slide, plan[i, , drop = FALSE])
    testthat::expect_equal(dim(tile), c(224L, 224L, 3L))
  }
})

testthat::test_that("Virchow2 tiling requires trustworthy spacing evidence", {
  slide <- wsiTools:::wsi_mock_slide(width = 1000, height = 600, levels = 1)
  slide$properties <- list()
  testthat::expect_error(wsi_virchow2_tile_plan(slide), "trustworthy microscopy")
  testthat::expect_error(wsi_virchow2_tile_plan(slide, source_mpp_xy = c(0.25, 0.25)),
                         "spacing_evidence")
  grid <- wsi_virchow2_tile_plan(slide, source_mpp_xy = c(0.25, 0.25),
                                 spacing_evidence = "pinned dataset metadata")
  testthat::expect_equal(unique(grid$native_width), 448L)
  testthat::expect_equal(unique(grid$native_height), 448L)
  testthat::expect_error(wsi_virchow2_tile_plan(slide, source_mpp_xy = c(1, 1),
                                               spacing_evidence = "source metadata"),
                         "coarser")
})

testthat::test_that("Virchow2 region tiles have no overlaps or gaps", {
  slide <- wsiTools:::wsi_mock_slide(width = 1000, height = 600, levels = 1)
  slide$properties <- list()
  slide$properties[["ome.physical-size-x"]] <- "0.273812"
  slide$properties[["ome.physical-size-y"]] <- "0.273813"
  grid <- wsi_virchow2_tile_plan(slide,
                                 region = c(x = 20, y = 30, width = 800, height = 500))
  testthat::expect_equal(unique(grid$native_width), 409L)
  testthat::expect_equal(unique(grid$native_height), 409L)
  testthat::expect_equal(min(grid$x), 20L)
  testthat::expect_equal(min(grid$y), 30L)
  testthat::expect_equal(max(grid$x + grid$valid_width), 820L)
  testthat::expect_equal(max(grid$y + grid$valid_height), 530L)
  testthat::expect_error(wsi_virchow2_tile_plan(slide, max_tiles = 1), "max_tiles")
})

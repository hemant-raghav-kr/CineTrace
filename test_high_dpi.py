"""
CineTrace — High-DPI and Image Rendering Quality Tests
Validates Windows High-DPI awareness, display scaling conversions,
Amazon CDN poster URL upgrading, aspect-ratio-preserving Lanczos downsampling,
and non-degraded multi-resolution master caching.
"""

import io
import unittest
from unittest.mock import patch, MagicMock
from PIL import Image

import dpi_helper
import image_loader


class TestDpiHelper(unittest.TestCase):
    """Test DPI awareness, display metrics, and pixel scaling calculations."""

    def test_enable_high_dpi_awareness_safe_call(self):
        """Calling enable_high_dpi_awareness returns boolean without crashing."""
        res = dpi_helper.enable_high_dpi_awareness()
        self.assertIsInstance(res, bool)

    def test_get_dpi_returns_valid_value(self):
        """get_dpi returns standard DPI (96) or higher on High-DPI monitors."""
        dpi = dpi_helper.get_dpi()
        self.assertIsInstance(dpi, int)
        self.assertGreaterEqual(dpi, 96)

    def test_get_dpi_scale_baseline(self):
        """get_dpi_scale computes accurate factor relative to 96 DPI baseline."""
        self.assertEqual(dpi_helper.get_dpi_scale(dpi=96), 1.0)
        self.assertEqual(dpi_helper.get_dpi_scale(dpi=120), 1.25)
        self.assertEqual(dpi_helper.get_dpi_scale(dpi=144), 1.5)
        self.assertEqual(dpi_helper.get_dpi_scale(dpi=192), 2.0)

    def test_scale_px_across_scaling_factors(self):
        """scale_px calculates exact pixel values without clipping or truncation."""
        # 100% scale (96 DPI)
        self.assertEqual(dpi_helper.scale_px(140, scale=1.0), 140)
        self.assertEqual(dpi_helper.scale_px(195, scale=1.0), 195)
        self.assertEqual(dpi_helper.scale_px(230, scale=1.0), 230)
        self.assertEqual(dpi_helper.scale_px(1260, scale=1.0), 1260)

        # 125% scale (120 DPI)
        self.assertEqual(dpi_helper.scale_px(140, scale=1.25), 175)
        self.assertEqual(dpi_helper.scale_px(195, scale=1.25), 244)
        self.assertEqual(dpi_helper.scale_px(230, scale=1.25), 288)

        # 150% scale (144 DPI)
        self.assertEqual(dpi_helper.scale_px(140, scale=1.5), 210)
        self.assertEqual(dpi_helper.scale_px(195, scale=1.5), 292)
        self.assertEqual(dpi_helper.scale_px(230, scale=1.5), 345)
        self.assertEqual(dpi_helper.scale_px(1260, scale=1.5), 1890)

        # 200% scale (192 DPI, 4K display)
        self.assertEqual(dpi_helper.scale_px(140, scale=2.0), 280)
        self.assertEqual(dpi_helper.scale_px(195, scale=2.0), 390)
        self.assertEqual(dpi_helper.scale_px(230, scale=2.0), 460)

        # Non-positive values remain untouched
        self.assertEqual(dpi_helper.scale_px(0, scale=1.5), 0)
        self.assertEqual(dpi_helper.scale_px(-5, scale=1.5), -5)

    def test_configure_tk_scaling_calculation(self):
        """configure_tk_scaling sets root scaling factor equal to dpi / 72.0."""
        mock_root = MagicMock()
        with patch.object(dpi_helper, "get_dpi", return_value=144):
            factor = dpi_helper.configure_tk_scaling(mock_root)
            self.assertAlmostEqual(factor, 2.0, places=3)
            mock_root.tk.call.assert_called_with("tk", "scaling", 2.0)


class TestPosterUrlUpgrading(unittest.TestCase):
    """Test Amazon CDN poster URL resolution upgrade rules."""

    def test_upgrade_standard_omdb_amazon_url(self):
        """Standard _SX300 thumbnail URL upgrades to high-res _SX900."""
        url = "https://m.media-amazon.com/images/M/MV5BMjAxMzY3NjcxNF5BMl5BanBnXkFtZTcwNTI5OTM0Mw@@._V1_SX300.jpg"
        upgraded = image_loader.get_high_res_poster_url(url)
        self.assertEqual(
            upgraded,
            "https://m.media-amazon.com/images/M/MV5BMjAxMzY3NjcxNF5BMl5BanBnXkFtZTcwNTI5OTM0Mw@@._V1_SX900.jpg"
        )

    def test_upgrade_cropped_and_compressed_amazon_url(self):
        """Complex Amazon preview URL upgrades to clean _SX900 master."""
        url = "https://m.media-amazon.com/images/M/MV5BMDAyY2FhYjctNDc5OS00MDNlLThiMGUtY2UxYWVkNGY2ZjljXkEyXkFqcGc@._V1_QL75_UX380_CR0,4,380,562_.jpg"
        upgraded = image_loader.get_high_res_poster_url(url)
        self.assertEqual(
            upgraded,
            "https://m.media-amazon.com/images/M/MV5BMDAyY2FhYjctNDc5OS00MDNlLThiMGUtY2UxYWVkNGY2ZjljXkEyXkFqcGc@._V1_SX900.jpg"
        )

    def test_upgrade_ia_media_imdb_url(self):
        """ia.media-imdb.com URL variant upgrades to _SX900."""
        url = "https://ia.media-imdb.com/images/M/MV5BNGEwMTQxZTQt._V1_SY500.jpg"
        upgraded = image_loader.get_high_res_poster_url(url)
        self.assertEqual(upgraded, "https://ia.media-imdb.com/images/M/MV5BNGEwMTQxZTQt._V1_SX900.jpg")

    def test_non_amazon_url_preserved(self):
        """Non-Amazon URLs are left unaltered."""
        ext_url = "https://example.com/posters/movie_poster.jpg"
        self.assertEqual(image_loader.get_high_res_poster_url(ext_url), ext_url)

    def test_na_and_empty_url_handling(self):
        """Missing or 'N/A' URLs are safely preserved without errors."""
        self.assertEqual(image_loader.get_high_res_poster_url("N/A"), "N/A")
        self.assertEqual(image_loader.get_high_res_poster_url(""), "")
        self.assertIsNone(image_loader.get_high_res_poster_url(None))


class TestImageResamplingAndDownscaling(unittest.TestCase):
    """Test aspect-ratio preserving Lanczos downsampling and placeholder generation."""

    def test_fit_and_pad_preserves_aspect_ratio(self):
        """fit_and_pad_image resizes image proportionally and centers inside canvas."""
        # Create a 2:3 high-res poster (600 x 900)
        source = Image.new("RGBA", (600, 900), (255, 0, 0, 255))
        fitted = image_loader.fit_and_pad_image(source, 210, 292)

        # Output canvas has requested dimensions
        self.assertEqual(fitted.size, (210, 292))

        # Check background padding color
        bg_pixel = fitted.getpixel((0, 0))
        self.assertEqual(bg_pixel, image_loader.CARD_BG_COLOR)

        # Check centered movie poster pixel
        center_pixel = fitted.getpixel((105, 146))
        self.assertEqual(center_pixel, (255, 0, 0, 255))

    def test_fit_and_pad_panoramic_image(self):
        """fit_and_pad_image handles wide images with vertical pillarboxing."""
        wide_source = Image.new("RGBA", (1000, 500), (0, 255, 0, 255))
        fitted = image_loader.fit_and_pad_image(wide_source, 200, 200)

        self.assertEqual(fitted.size, (200, 200))
        # Top and bottom are padded
        self.assertEqual(fitted.getpixel((100, 5)), image_loader.CARD_BG_COLOR)
        # Center is the green image
        self.assertEqual(fitted.getpixel((100, 100)), (0, 255, 0, 255))

    def test_create_placeholder_scales_dimensions(self):
        """create_placeholder_image generates exact target size at High-DPI."""
        p_normal = image_loader.create_placeholder_image(140, 195, "Inception")
        self.assertEqual(p_normal.size, (140, 195))

        p_hidpi = image_loader.create_placeholder_image(210, 292, "Inception")
        self.assertEqual(p_hidpi.size, (210, 292))


class TestMasterImageCachingPipeline(unittest.TestCase):
    """Test master PIL caching to prevent repeated downscaling quality degradation."""

    def setUp(self):
        image_loader.clear_image_caches()

    def tearDown(self):
        image_loader.clear_image_caches()

    def test_master_image_cached_and_reused_without_re_download(self):
        """Multiple size requests (card and dialog) reuse master original from _PIL_CACHE."""
        url = "https://m.media-amazon.com/images/M/MV5BMTEST._V1_SX300.jpg"

        # Generate a simulated high-res master (900 x 1350)
        master_pil = Image.new("RGBA", (900, 1350), (100, 150, 200, 255))
        master_bytes = io.BytesIO()
        master_pil.save(master_bytes, format="PNG")
        master_bytes_data = master_bytes.getvalue()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = master_bytes_data

        with patch("requests.get", return_value=mock_resp) as mock_get:
            # First request: Card at 150% (210 x 292)
            card_img = image_loader.fetch_and_fit_pil(url, 210, 292)
            self.assertIsNotNone(card_img)
            self.assertEqual(card_img.size, (210, 292))

            # Master should now be in _PIL_CACHE under original url
            self.assertIn(url, image_loader._PIL_CACHE)
            cached_master = image_loader._PIL_CACHE[url]
            self.assertEqual(cached_master.size, (900, 1350))

            # Second request: Details dialog at 150% (300 x 435)
            dialog_img = image_loader.fetch_and_fit_pil(url, 300, 435)
            self.assertIsNotNone(dialog_img)
            self.assertEqual(dialog_img.size, (300, 435))

            # requests.get should only have been called ONCE because master is cached!
            self.assertEqual(mock_get.call_count, 1)

            # Upgraded URL was requested
            requested_url = mock_get.call_args[0][0]
            self.assertEqual(
                requested_url,
                "https://m.media-amazon.com/images/M/MV5BMTEST._V1_SX900.jpg"
            )

            # Both fitted variants are distinct and cached in _FITTED_PIL_CACHE
            self.assertIn((url, 210, 292), image_loader._FITTED_PIL_CACHE)
            self.assertIn((url, 300, 435), image_loader._FITTED_PIL_CACHE)


if __name__ == "__main__":
    unittest.main()

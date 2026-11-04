import re
import unittest
from pathlib import Path

HTML = Path(__file__).with_name('portal.html').read_text()


class ThemeContractTest(unittest.TestCase):
    def test_shared_typography_and_operator_purple_theme(self):
        self.assertIn('--ui:Inter,', HTML)
        self.assertIn('font:15px/1.5 var(--ui)', HTML)
        self.assertIn('--purple:#6557d2', HTML)
        self.assertNotIn('font-family:"Microsoft YaHei"', HTML)

    def test_both_business_frames_and_readonly_notice_are_preserved(self):
        self.assertIn('src="/model/"', HTML)
        self.assertIn('data-src="/operator/"', HTML)
        self.assertNotIn('只读预览', HTML)
        self.assertNotIn('内网预览', HTML)


if __name__ == '__main__':
    unittest.main()

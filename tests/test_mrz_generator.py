import random
import unittest

from src.mrz_generator import check_digit, make_td3


class MRZGeneratorTests(unittest.TestCase):
    def test_check_digit_reference(self):
        self.assertEqual(check_digit("L898902C3"), "6")

    def test_td3_has_fixed_lengths(self):
        rec = make_td3(random.Random(123))
        self.assertEqual(len(rec.line1), 44)
        self.assertEqual(len(rec.line2), 44)
        self.assertTrue(set(rec.line1).issubset(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")))
        self.assertTrue(set(rec.line2).issubset(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")))


if __name__ == "__main__":
    unittest.main()

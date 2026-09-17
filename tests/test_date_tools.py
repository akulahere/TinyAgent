from datetime import date
import unittest
from unittest.mock import patch

from toolbox import days_between, today
from tools import tool_to_schema


class DateToolTests(unittest.TestCase):
    def test_today_reads_the_host_date_each_time(self):
        with patch("toolbox.date") as clock:
            clock.today.side_effect = [date(2026, 9, 17), date(2026, 9, 18)]
            self.assertEqual(today(), "2026-09-17")
            self.assertEqual(today(), "2026-09-18")

    def test_date_difference_boundary_leap_year_and_direction(self):
        self.assertEqual(days_between("2026-09-17", "2026-09-17"), 0)
        self.assertEqual(days_between("2026-12-31", "2027-01-01"), 1)
        self.assertEqual(days_between("2024-02-28", "2024-03-01"), 2)
        self.assertEqual(days_between("2025-02-28", "2025-03-01"), 1)
        self.assertEqual(days_between("2024-03-01", "2024-02-28"), -2)

    def test_invalid_dates_are_rejected(self):
        for a, b in (("not a date", "2030-12-31"), ("2026-09-17", "2030-02-30")):
            with self.assertRaises(ValueError):
                days_between(a, b)

    def test_native_schemas_include_zero_argument_clock_and_date_strings(self):
        self.assertEqual(tool_to_schema(today)["function"]["parameters"]["properties"], {})
        schema = tool_to_schema(days_between)["function"]["parameters"]
        self.assertEqual(schema["required"], ["a", "b"])
        self.assertEqual(schema["properties"], {"a": {"type": "string"}, "b": {"type": "string"}})

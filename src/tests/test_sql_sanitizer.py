import unittest
from sanitizer.sql_sanitizer import SQLSanitizer


class TestSQLSanitizer(unittest.TestCase):
    """Unit tests for the SQLSanitizer class."""

    def setUp(self):
        """Create one SQLSanitizer instance for all tests."""
        self.sanitizer = SQLSanitizer()

    def test_sanitize_queries(self):
        """Tests multiple SQL sanitization cases."""

        test_cases = [
            (
                """SELECT Notes FROM table WHERE Current slogan = SOUTH AUSTRALIA AND Next slogan < AFRICA OR blabla blabla > 2""",
                {"id": "1-1000181-1", "header": ['Description', "Next slogan", "Current slogan"]},
                """SELECT `Notes` FROM table_1_1000181_1 WHERE `Current slogan` = "SOUTH AUSTRALIA" AND `Next slogan` < "AFRICA" OR `blabla blabla` > 2;"""
            ),
            (
                """SELECT AVG `% of pop.` FROM table WHERE `Country (or dependent territory)` = "tunisia" AND `Average relative annual growth (%)` < "1.03";""",
                {"id": "2-12101133-1", "header": ['% of pop.', "Average relative annual growth (%)", "Country (or dependent territory)"]},
                """SELECT AVG(`% of pop.`) FROM table_2_12101133_1 WHERE `Country (or dependent territory)` = "tunisia" AND `Average relative annual growth (%)` < "1.03";"""
            ),
            (
                """SELECT COUNT Fleet Series (Quantity) FROM table WHERE Fuel Propulsion = CNG""",
                {"id": "1-1000181-1", "header": ['Fleet Series (Quantity)', 'Description']},
                """SELECT COUNT(`Fleet Series (Quantity)`) FROM table_1_1000181_1 WHERE `Fuel Propulsion` = "CNG";"""
            ),
            (
                """SELECT COUNT MAX Fleet Series (Quantity) FROM table WHERE Fuel Propulsion = CNG""",
                {"id": "1-1000181-1", "header": ['Fleet Series (Quantity)', 'Description']},
                """SELECT COUNT(MAX(`Fleet Series (Quantity)`)) FROM table_1_1000181_1 WHERE `Fuel Propulsion` = "CNG";"""
            ),
            (
                """SELECT MIN `GDP per cap. (2003, in €)` FROM table WHERE `Province` = "Friesland";""",
                {"id": "1-1000181-1", "header": ['GDP per cap. (2003, in €)', 'Description']},
                """SELECT MIN(`GDP per cap. (2003, in €)`) FROM table_1_1000181_1 WHERE `Province` = "Friesland";"""
            ),
            (
                """SELECT MAX `Avg. emission per km 2 of its land (tons)` FROM table WHERE `Country` = "India";""",
                {"id": "1-1000181-1", "header": ['Avg. emission per km 2 of its land (tons)', 'Description']},
                """SELECT MAX(`Avg. emission per km 2 of its land (tons)`) FROM table_1_1000181_1 WHERE `Country` = "India";"""
            ),
            (
                """SELECT `Extroverted, Relationship-Oriented` FROM table WHERE `Extroverted, Task-Oriented` = "Director";""",
                {"id": "1-1000181-1", "header": ['Extroverted, Relationship-Oriented', 'Extroverted, Task-Oriented', 'Description']},
                """SELECT `Extroverted, Relationship-Oriented` FROM table_1_1000181_1 WHERE `Extroverted, Task-Oriented` = "Director";"""
            ),
            (
                """SELECT `Max Gross Weight` FROM table WHERE `Aircraft` = "Robinson R-22";""",
                {"id": "1-1000181-1", "header": ['Aircraft', 'Description', 'Max Gross Weight', 'Total disk area', 'Max disk Loading']},
                """SELECT `Max Gross Weight` FROM table_1_1000181_1 WHERE `Aircraft` = "Robinson R-22";"""
            ),
            (
                """SELECT AVG Events FROM table WHERE Top-5 < 1 AND Top-25 > 1;""",
                {"id": "1-1000181-1", "header": ['Events', 'Top-5', 'Top-25']},
                """SELECT AVG(`Events`) FROM table_1_1000181_1 WHERE `Top-5` < 1 AND `Top-25` > 1;"""
            ),
            (
                """SELECT COUNT Singles W–L FROM table WHERE Doubles W–L = 11–14;""",
                {"id": "1-1000181-1", "header": ['Singles W–L', 'Doubles W–L']},
                """SELECT COUNT(`Singles W–L`) FROM table_1_1000181_1 WHERE `Doubles W–L` = "11–14";"""
            ),
            (
                """SELECT State FROM table WHERE Height = 5'4" AND Hometown = Lexington, Ky;""",
                {"id": "1-1000181-1", "header": ['State', 'Height', 'Hometown']},
                """SELECT `State` FROM table_1_1000181_1 WHERE `Height` = "5'4"" AND `Hometown` = "Lexington, Ky";"""
            ),
            (
                """SELECT Men's doubles FROM table WHERE Men's singles = no competition;""",
                {"id": "1-1000181-1", "header": ["Men's doubles", "Men's singles"]},
                """SELECT `Men's doubles` FROM table_1_1000181_1 WHERE `Men's singles` = "no competition";"""
            ),
            (
                """SELECT Year FROM table WHERE Pts. > 0 AND Entrant = ecurie bleue;""",
                {"id": "1-1000181-1", "header": ["Year", "Pts.", "Entrant"]},
                """SELECT `Year` FROM table_1_1000181_1 WHERE `Pts.` > 0 AND `Entrant` = "ecurie bleue";"""
            ),
            (
                """SELECT No. FROM table_1_10015132_16 WHERE Years in Toronto = 1995-96;""",
                {"id": "1-10015132-16", "header": ["No.", "Years in Toronto"]},
                """SELECT `No.` FROM table_1_10015132_16 WHERE `Years in Toronto` = "1995-96";"""
            ),
        ]

        for i, (query, metadata, expected) in enumerate(test_cases, start=1):
            with self.subTest(test=i):
                result = self.sanitizer.sanitize(query, metadata, verbose=0)
                self.assertEqual(
                    result, expected,
                    f"Failed test {i}\nGot:\n{result}\nExpected:\n{expected}"
                )


if __name__ == "__main__":
    unittest.main()
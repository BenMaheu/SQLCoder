from sanitizer.sql_sanitizer import SQLSanitizer


def test_sql_sanitizer(sanitizer: SQLSanitizer):
    test_case1 = """SELECT Notes FROM table WHERE Current slogan = SOUTH AUSTRALIA AND Next slogan < AFRICA OR blabla blabla > 2"""
    test_case2 = """SELECT AVG `% of pop.` FROM table WHERE `Country (or dependent territory)` = "tunisia" AND `Average relative annual growth (%)` < "1.03";"""
    test_case3 = """SELECT COUNT Fleet Series (Quantity) FROM table WHERE Fuel Propulsion = CNG"""
    test_case4 = """SELECT COUNT MAX Fleet Series (Quantity) FROM table WHERE Fuel Propulsion = CNG"""
    test_case5 = """SELECT MIN `GDP per cap. (2003, in €)` FROM table WHERE `Province` = "Friesland";"""
    test_case6 = """SELECT MAX `Avg. emission per km 2 of its land (tons)` FROM table WHERE `Country` = "India";"""
    test_case7 = """SELECT `Extroverted, Relationship-Oriented` FROM table WHERE `Extroverted, Task-Oriented` = "Director";"""
    test_case8 = """SELECT `Max Gross Weight` FROM table WHERE `Aircraft` = "Robinson R-22";"""
    test_case9 = """SELECT MIN(`Left`) FROM table_1_11658094_3 WHERE `Institution` = "Longwood University";"""  # Check for re-sanitization
    test_case10 = """SELECT AVG Events FROM table WHERE Top-5 < 1 AND Top-25 > 1;"""  # Check for hyphens in col names
    test_case11 = """SELECT COUNT Singles W–L FROM table WHERE Doubles W–L = 11–14;"""  # Check for –— hyphen (not ascii hyphen)
    test_case12 = """SELECT State FROM table WHERE Height = 5'4" AND Hometown = Lexington, Ky;"""  # Check for –— hyphen (not ascii hyphen)
    test_case13 = """SELECT Men's doubles FROM table WHERE Men's singles = no competition;"""  # Check "'"
    test_case14 = """SELECT Year FROM table WHERE Pts. > 0 AND Entrant = ecurie bleue;"""  # Check "."

    expected1 = """SELECT `Notes` FROM table_1_1000181_1 WHERE `Current slogan` = "SOUTH AUSTRALIA" AND `Next slogan` < "AFRICA" OR `blabla blabla` > 2;"""
    expected2 = """SELECT AVG(`% of pop.`) FROM table_2_12101133_1 WHERE `Country (or dependent territory)` = "tunisia" AND `Average relative annual growth (%)` < "1.03";"""  # i= 35177
    expected3 = """SELECT COUNT(`Fleet Series (Quantity)`) FROM table_1_1000181_1 WHERE `Fuel Propulsion` = "CNG";"""
    expected4 = """SELECT COUNT(MAX(`Fleet Series (Quantity)`)) FROM table_1_1000181_1 WHERE `Fuel Propulsion` = "CNG";"""
    expected5 = """SELECT MIN(`GDP per cap. (2003, in €)`) FROM table_1_1000181_1 WHERE `Province` = "Friesland";"""
    expected6 = """SELECT MAX(`Avg. emission per km 2 of its land (tons)`) FROM table_1_1000181_1 WHERE `Country` = "India";"""
    expected7 = """SELECT `Extroverted, Relationship-Oriented` FROM table_1_1000181_1 WHERE `Extroverted, Task-Oriented` = "Director";"""
    expected8 = """SELECT `Max Gross Weight` FROM table_1_1000181_1 WHERE `Aircraft` = "Robinson R-22";"""
    expected9 = """SELECT MIN(`Left`) FROM table_1_11658094_3 WHERE `Institution` = "Longwood University";"""
    expected10 = """SELECT AVG(`Events`) FROM table_1_1000181_1 WHERE `Top-5` < 1 AND `Top-25` > 1;"""
    expected11 = """SELECT COUNT(`Singles W–L`) FROM table_1_1000181_1 WHERE `Doubles W–L` = "11–14";"""
    expected12 = """SELECT `State` FROM table_1_1000181_1 WHERE `Height` = "5'4"" AND `Hometown` = "Lexington, Ky";"""
    expected13 = """SELECT `Men's doubles` FROM table_1_1000181_1 WHERE `Men's singles` = "no competition";"""
    expected14 = """SELECT `Year` FROM table_1_1000181_1 WHERE `Pts.` > 0 AND `Entrant` = "ecurie bleue";"""

    out1 = sanitizer.sanitize(test_case1,
                              {"id": "1-1000181-1", "header": ['Description', "Next slogan", "Current slogan"]},
                              verbose=1)
    out2 = sanitizer.sanitize(test_case2, {"id": "2-12101133-1",
                                           "header": ['% of pop.', "Average relative annual growth (%)",
                                                      "Country (or dependent territory)"]}, verbose=1)
    out3 = sanitizer.sanitize(test_case3, {"id": "1-1000181-1", "header": ['Fleet Series (Quantity)', 'Description']},
                              verbose=1)
    out4 = sanitizer.sanitize(test_case4, {"id": "1-1000181-1", "header": ['Fleet Series (Quantity)', 'Description']},
                              verbose=1)
    out5 = sanitizer.sanitize(test_case5, {"id": "1-1000181-1", "header": ['GDP per cap. (2003, in €)', 'Description']},
                              verbose=1)
    out6 = sanitizer.sanitize(test_case6, {"id": "1-1000181-1",
                                           "header": ['Avg. emission per km 2 of its land (tons)', 'Description']},
                              verbose=1)
    out7 = sanitizer.sanitize(test_case7, {"id": "1-1000181-1", "header": ['Extroverted, Relationship-Oriented',
                                                                           'Extroverted, Task-Oriented',
                                                                           'Description']}, verbose=1)
    out8 = sanitizer.sanitize(test_case8, {"id": "1-1000181-1",
                                           "header": ['Aircraft', 'Description', 'Max Gross Weight', 'Total disk area',
                                                      'Max disk Loading']}, verbose=1)
    out9 = sanitizer.sanitize(test_case9, {"id": "1-11658094-3", "header": ['Left', 'Max disk Loading']}, verbose=1)
    out10 = sanitizer.sanitize(test_case10, {"id": "1-1000181-1", "header": ['Events', 'Top-5', 'Top-25']}, verbose=1)
    out11 = sanitizer.sanitize(test_case11, {"id": "1-1000181-1", "header": ['Events', 'Top-5', 'Top-25']}, verbose=1)
    out12 = sanitizer.sanitize(test_case12, {"id": "1-1000181-1", "header": ['Events', 'Top-5', 'Top-25']}, verbose=1)
    out13 = sanitizer.sanitize(test_case13, {"id": "1-1000181-1", "header": ["Men's doubles", "Men's singles"]},
                               verbose=1)
    out14 = sanitizer.sanitize(test_case14, {"id": "1-1000181-1", "header": ["Year", "Pts.", "Entrant"]}, verbose=1)

    assert out1 == expected1, "Failed test 1 \nGot sanitized:\n" + out1 + "\n" + "\nExpected:\n" + expected1
    assert out2 == expected2, "Failed test 2 \nGot sanitized:\n" + out2 + "\n" + "\nExpected:\n" + expected2
    assert out3 == expected3, "Failed test 3 \nGot sanitized:\n" + out3 + "\n" + "\nExpected:\n" + expected3
    assert out4 == expected4, "Failed test 4 \nGot sanitized:\n" + out4 + "\n" + "\nExpected:\n" + expected4
    assert out5 == expected5, "Failed test 5 \nGot sanitized:\n" + out5 + "\n" + "\nExpected:\n" + expected5
    assert out6 == expected6, "Failed test 6 \nGot sanitized:\n" + out6 + "\n" + "\nExpected:\n" + expected6
    assert out7 == expected7, "Failed test 7 \nGot sanitized:\n" + out7 + "\n" + "\nExpected:\n" + expected7
    assert out8 == expected8, "Failed test 8 \nGot sanitized:\n" + out8 + "\n" + "\nExpected:\n" + expected8
    assert out9 == expected9, "Failed test 9 \nGot sanitized:\n" + out9 + "\n" + "\nExpected:\n" + expected9
    assert out10 == expected10, "Failed test 10 \nGot sanitized:\n" + out10 + "\n" + "\nExpected:\n" + expected10
    assert out11 == expected11, "Failed test 11 \nGot sanitized:\n" + out11 + "\n" + "\nExpected:\n" + expected11
    assert out12 == expected12, "Failed test 12 \nGot sanitized:\n" + out12 + "\n" + "\nExpected:\n" + expected12
    assert out13 == expected13, "Failed test 13 \nGot sanitized:\n" + out13 + "\n" + "\nExpected:\n" + expected13
    assert out14 == expected14, "Failed test 14 \nGot sanitized:\n" + out14 + "\n" + "\nExpected:\n" + expected14
    print("\n" + "=" * 50)
    print("\n\nAll checks succeeded for SQL Sanitizer")


if __name__ == '__main__':
    sanitizer = SQLSanitizer()
    test_sql_sanitizer(sanitizer)
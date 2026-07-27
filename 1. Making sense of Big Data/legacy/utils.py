import numpy as np

def validate_sales_results(item_revenues, total_revenue):
    """
    Validates the results of the sales calculation exercise with user-friendly feedback.
    """
    # Expected values
    expected_prices = np.array([10, 15, 20])
    expected_quantity = np.array([5, 8, 3])

    # Recalculate the expected results
    expected_item_revenues = expected_prices * expected_quantity
    expected_total_revenue = np.sum(expected_item_revenues)

    # Validate each step and provide specific feedback
    if not np.allclose(item_revenues, expected_item_revenues):
        print("❌ The `item_revenues` calculation seems incorrect.")
        return

    if not np.allclose(total_revenue, expected_total_revenue):
        print("❌ The `total_revenue` calculation seems incorrect.")
        return

    print("✅ All calculations are correct! Well done.")

def validate_titanic_stats(average_age, max_fare, total_survivors, senior_citizens=None):
    """
    Validates the results of the Titanic statistics calculation exercise.
    """
    expected_avg_age = 29.69911764705882
    expected_max_fare = 512.3292
    expected_survivors = 342
    expected_senior_citizens = 22

    if not np.isclose(average_age, expected_avg_age):
        print("❌ The `average_age` calculation seems incorrect.")
        return

    if not np.isclose(max_fare, expected_max_fare):
        print("❌ The `max_fare` calculation seems incorrect.")
        return

    if total_survivors != expected_survivors:
        print("❌ The `total_survivors` calculation seems incorrect.")
        return

    if senior_citizens is not None and senior_citizens != expected_senior_citizens:
        print("❌ The `senior_citizens` calculation seems incorrect.")
        return

    print("✅ All Titanic statistics are correct! Awesome job.")

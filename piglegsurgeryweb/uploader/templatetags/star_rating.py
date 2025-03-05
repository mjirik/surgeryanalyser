# myapp/templatetags/star_rating.py
from django import template

register = template.Library()


@register.filter
def star_rating(score):
    """
    Converts a numeric score (0-100) to a star rating out of 5 stars.
    For example, a score of 80 returns 4 filled stars and 1 empty star.
    """
    try:
        score = float(score)
    except (TypeError, ValueError):
        return ""

    # Calculate how many full stars (each star represents 20 points)
    full_stars = int(score / 20)

    # Ensure not to exceed 5 stars
    if full_stars > 5:
        full_stars = 5

    # Calculate remaining stars as empty stars
    empty_stars = 5 - full_stars

    # You can use Unicode stars or use images/icons as needed.
    return "★" * full_stars + "☆" * empty_stars

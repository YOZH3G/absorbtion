def combine_fractions(first, second):
    """Return the combined relative change for two independent fractions."""
    return (1 + first) * (1 + second) - 1


def calculate_xog(gg, xg, gog, k_xg=0.0, k_gg=0.0):
    """Calculate lean-gas concentration after composition and flow changes."""
    return (gg * (1 + k_gg) * xg * (1 + k_xg)) / gog


def calculate_xna(ga, gna, xa, k_xa=0.0, k_ga=0.0):
    """Calculate rich-absorbent concentration after composition and flow changes."""
    return (ga * (1 + k_ga) / gna) * xa * (1 + k_xa)

import Levenshtein


def compute_error_rate(s1: str, s2: str, indel_cost: float = 1, replace_cost: float = 1) -> float:
    # If reference and sequence are empty, return 0
    if not s1 and not s2:
        return 0

    # Get edit operations
    ops = Levenshtein.editops(s1, s2)

    # Calculate cost
    cost = 0.0
    for op in ops:
        if op[0] in ["delete", "insert"]:
            cost += indel_cost
        elif op[0] == "replace":
            cost += replace_cost

    # Normalize cost
    return cost / max(len(s2), len(s1))

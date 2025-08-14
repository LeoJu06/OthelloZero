
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:1e}", "s"
    elif seconds < 3600:
        return f"{seconds / 60:1e}", "min"
    elif seconds < 86400:
        return f"{seconds / 3600:.1e}", "h"
    elif seconds < 31_536_000:
        return f"{seconds / 86400:1e}", "d"
    else:
        return f"{seconds / 31_536_000:1e}", "a"

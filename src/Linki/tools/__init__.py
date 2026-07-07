__all__ = ["build_read_only_tools", "build_tools"]


def build_tools(state):
    from Linki.tools.registry import build_tools as _build_tools

    return _build_tools(state)


def build_read_only_tools(state):
    from Linki.tools.registry import build_read_only_tools as _build_read_only_tools

    return _build_read_only_tools(state)

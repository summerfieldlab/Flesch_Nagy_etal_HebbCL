import os
import sys

root_path = os.path.realpath("./")
sys.path.append(root_path)

import pytest  # noqa E402
from hebbcl.parameters import parser  # noqa E402


def parse_args(args):
    return parser.parse_args(args)


class TestArgparse:
    def test_none_str(self):
        """tests whether "none" arg correctly replaced with None"""
        args = parse_args(["--gating=None"])
        assert args.gating is None

        args = parse_args(["--gating=none"])
        assert args.gating is None

        args = parse_args(["--gating=oja_ctx"])
        assert args.gating == "oja_ctx"


if __name__ == "__main__":
    pytest.main([__file__])

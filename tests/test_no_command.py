import io
import sys
import unittest
from unittest.mock import patch

from bioclip.__main__ import main, parse_args


class TestNoCommand(unittest.TestCase):
    def test_parse_args_without_command(self):
        args = parse_args([])
        self.assertIsNone(args.command)

    def test_no_command_prints_help_and_returns_error(self):
        # Running `bioclip` with no subcommand must show usage and exit
        # nonzero instead of silently succeeding.
        buf = io.BytesIO()
        fake_stdout = io.TextIOWrapper(buf, encoding='utf-8')
        with patch.object(sys, 'argv', ['bioclip']):
            with patch.object(sys, 'stdout', fake_stdout):
                result = main()
        fake_stdout.flush()
        self.assertEqual(result, 2)
        self.assertIn('usage', buf.getvalue().decode('utf-8').lower())


if __name__ == '__main__':
    unittest.main()

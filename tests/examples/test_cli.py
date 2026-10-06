from __future__ import annotations

import subprocess
import sys
import unittest


class CliTest(unittest.TestCase):
    def test_help_exposes_manifest_input(self):
        process = subprocess.run(
            [sys.executable, '-m', 'rebar_inverse', '--help'],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('--manifest', process.stdout)
        self.assertIn('--mesh-label-source', process.stdout)


if __name__ == '__main__':
    unittest.main()

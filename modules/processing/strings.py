# Copyright (C) 2010-2013 Claudio Guarnieri.
# Copyright (C) 2014-2016 Cuckoo Foundation.
# Copyright (C) 2020-2021 PowerLZY.
# This file is part of Cuckoo Sandbox - http://www.cuckoosandbox.org
# See the file 'docs/LICENSE' for copying permission.

import os.path
import re
import chardet

from lib.cuckoo.common.abstracts import Processing
from lib.cuckoo.common.exceptions import CuckooProcessingError

class Strings(Processing):
    """Extract strings from analyzed file."""

    def run(self):
        """
        Run extract of printable strings.

        :return: list of printable strings.
        """
        self.key = "strings"
        strings = []

        if self.task["category"] == "file":
            if not os.path.exists(self.file_path):
                raise CuckooProcessingError("Sample file doesn't exist: \"%s\"" % self.file_path)
            try:
                data = open(self.file_path, "rb").read()
            except (IOError, OSError) as e:
                raise CuckooProcessingError("Error opening file %s" % e)
            strings = re.findall(b"[\x1f-\x7e]{6,}", data)
            strings = [bs.decode('utf-8') for bs in strings]
            strings += [str(ws.decode("utf-16le")) for ws in re.findall(b"(?:[\x1f-\x7e][\x00]){6,}", data)]

        return strings

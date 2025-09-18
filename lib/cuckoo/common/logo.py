# Copyright (C) 2010-2013 Claudio Guarnieri.
# Copyright (C) 2014-2016 Cuckoo Foundation.
# Copyright (C) 2020-2021 PowerLZY.
# This file is part of Cuckoo Sandbox - http://www.cuckoosandbox.org
# See the file 'docs/LICENSE' for copying permission.

import sys
import random

from lib.cuckoo.common.colors import color, yellow
from lib.cuckoo.common.constants import CUCKOO_VERSION

def logo():
    """
    Bold-Falcon asciiarts.

    :return: asciiarts array.
    """
    logos = []

    logos.append("""
        _|_|_|                                        _|          _|    _|                                _|        
        _|        _|_|_|  _|_|      _|_|_|  _|  _|_|  _|_|_|_|      _|    _|    _|_|_|  _|      _|      _|  _|  _|    
        _|_|    _|    _|    _|  _|    _|  _|_|        _|          _|_|_|_|  _|    _|  _|      _|      _|  _|_|      
            _|  _|    _|    _|  _|    _|  _|          _|          _|    _|  _|    _|    _|  _|  _|  _|    _|  _|    
        _|_|_|    _|    _|    _|    _|_|_|  _|            _|_|      _|    _|    _|_|_|      _|      _|      _|    _|  
    """)


    print(color(random.choice(logos), random.randrange(31, 37)))
    print
    print(" Smart Hawk %s" % yellow(CUCKOO_VERSION))
    print(" hawk.conimi.com ")
    print(" Copyright (c) 2020-2023")
    print
    sys.stdout.flush()

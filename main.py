# -*- coding: utf-8 -*-

from moduleLoading import Loaders as ldr

dataset = ldr.connectAndFetch("127.0.0.1", "mci_db", "root", "toor", "SELECT * FROM v5_mmse_pre")

print(dataset)

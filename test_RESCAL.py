import config
import models
import tensorflow as tf
import numpy as np
import json

con = config.Config()
con.set_in_path("./benchmarks/FB15K/")
con.set_test_flag(True)
con.set_work_threads(4)
con.set_dimension(50)
con.set_import_files("./res/model.vec.tf")
con.init()
con.set_model(models.RESCAL)
con.test()
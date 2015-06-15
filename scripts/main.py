# -*- coding: utf-8 -*-
import feature_extraction
import feature_extraction_real
import classifier
import datetime

timer = datetime.datetime.now()
feature_extraction.run(2500000)
feature_extraction_real.run()
classifier.run()
print("%s seconds passed" % (datetime.datetime.now() - timer).total_seconds())


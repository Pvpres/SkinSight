import splitfolders
import os

splitfolders.ratio(os.path.join(os.getcwd(), "usable_data"), output=os.path.join(os.getcwd(), "usable_data"), seed=42, ratio=(.8, .1, .1))

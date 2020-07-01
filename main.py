from preprocessing import *
from elections_results import *
from coalition_task_generative import *

if __name__ == '__main__':
    preprocess()
    predict_division_of_voters()
    find_steady_coalition()


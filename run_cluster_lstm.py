import sys, csv, math
from multiprocessing import Process
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sim.perf_model import *
from kleio.page_selector import *
from kleio.lstm import *

if __name__ == "__main__":
  
  # Command line arguments
  trace_dir = sys.argv[1]
  resdir = sys.argv[2]
  
  # Read trace
  app, app_label = 'cpd_10000', 'cpd'
  trace_file = trace_dir + 'trace_' + app + '.txt'
  prof = Profile(trace_file)
  prof.init()
  
  # Convert to per page access counts
  sim = PerfModel(prof, 'Fast:NearSlow', 'history', 0.2, 35650) # cori's frequency, so that less periods for RNN training
  sim.init()
  #sim.run()

  # Get misplaced pages eligible for RNNs.
  #page_selector = PageSelector(prof, 'Fast:NearSlow', '0.2', 35650, resdir + app_label + '_')
  #pages_misplaced = page_selector.get_misplaced_pages_sim()
  #pages_ordered = page_selector.get_ordered_pages(pages_misplaced)
  
  page_id_x = 593
  
  ### Make the RNN input
  # Step 1: take the page access count across periods
  cnts_x = prof.hmem.page_list[page_id_x].oracle_counts_binned_ep
  input = LSTM_input(cnts_x)
  
  # Step 2: Roll a window of history length over the periods
  history_length = 6 # periods
  input.timeseries_to_history_seq(history_length)
  
  # Step 3: Split into training, validation and test samples througout the epochs
  input.split_data(0.2)
  
  # Step 4: Bring input into format for RNN training
  num_classes = max(set(cnts_x)) + 1
  input.to_categor(num_classes)
  #input.prepare()
  
  ### Make the RNN model
  model = LSTM_model(input)

  model.create(256, 0.00001, 0, history_length, num_classes)
  
  model.train()
  model.infer()
#!/usr/bin/env python
from thetvdb import thetvdb
import argparse
import ConfigParser
import sys
import os
import kaa.metadata
import pylab as pl
from numpy import array,vstack,hstack,mgrid,c_
from sklearn import svm

#Global Arguments
version = '0.1'
args = None

#tv = thetvdb.TVShow('94571')

class Video:
  """Easy referencing of video classifications.
  
  """
  tv=1
  movie=0

class Classifier:
  def __init__(self):
    self.svc = svm.SVC(kernel="linear")
    self.__X = None
    self.__y = None
  
  def get_video_features(self,filename):
    """Gather the features of the given file and return a row to be used in the SVM.
    
    Parameter is the absolute filename
    """
    
    print_log_verbose("Gathering video features for "+str(filename))
    if os.path.exists(filename):
      info = kaa.metadata.parse(filename)
    else:
      print_error("get_video_features()","file cannot be found")
      return None
    print_log_verbose("Media type for: "+str(info.media))
    if info is not None and info.media == "MEDIA_AV":
      #gather features
      x1 = int(info.length)
      if args.verbose:
        print "x1 (video length): "+str(x1)
      return [x1]
    return None

  def gather_training_data(self,directory,classification):
    """Add to the current training set of data with the media files existing on the system. For each
    file, gather features to add it to the X array (a num_files X num_features array) and the
    classification in the y vector (a num_files length vector) matching 1 for tv and 0 for movies.
    
    There is no return value. This function builds on the current contents of X and y
    
    """
    
    #gather data in lists so we can bulk-add to the matrix
    #appending one row at a time to a matrix is costly
    X_rows = []
    y_rows = []
    
    for path,subdirs,files in os.walk(directory):
      for filename in files:
        absolute_path = os.path.join(path, filename)
        print_log_verbose("processing file "+filename)
        info = kaa.metadata.parse(absolute_path)
        #only process video files
        #documentation here: http://doc.freevo.org/api/kaa/metadata/usage.html#attributes-keys
        if info is not None and info.media == "MEDIA_AV":
          #gather features
          row = self.get_video_features(absolute_path)
          #add to training set
          if row is not None:
            print_log_verbose("adding row: "+str(row))
            X_rows.append(row)
            y_rows.append(classification)
    
    #now add the gathered data to the array
    if len(X_rows) > 0:
      print_log_verbose("X_rows: "+str(X_rows))
      print_log_verbose("y_rows: "+str(y_rows))
      if self.__X is None:
        self.__X = array(X_rows)
        self.__y = array(y_rows)
      else:
        self.__X = vstack((self.__X,X_rows))
        self.__y = hstack((self.__y,y_rows))
    

  def train(self):
    self.svc.fit(self.__X,self.__y)
  
  def classify(self,filename):
    """Classify the given file using the SVM.
    Return a classification from the Video class.
    
    """
    
    features = get_video_features(filename)
    if features is not None:
      print_log_verbose("classifying "+str(filename))
      print_log_verbose("features: "+str(features))
      return int(self.svc.predict([features])[0])
    else:
      return -1

  def plot_training_data(self):
    """Plot the training data to the screen to be used for troubleshooting.
    This was adapted from http://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html
    
    """
    print_log("X:\n"+str(self.__X))
    print_log("y:\n"+str(self.__y))

def print_error(section,message):
  print "ERROR: %s: %s" % (section,message)

def print_error_and_exit(section,message):
  print_error(section,message)
  sys.exit(1)

def print_log(message):
  print message

def print_log_verbose(message):
  if args.verbose:
    print_log(message)

def run():
  """Function to be run by the daemonized process.
  
  """
  while True:
    print "running"

def verify_config(config):
  """Verify that the essential parts of the configuration are provided in the ConfigParser object.
  Return False if an error was found.
  """
  ##General Section
  if config.has_section('GENERAL'):
    if config.has_option('GENERAL','watch_dir'):
      watch_dir=config.get('GENERAL','watch_dir')
    else:
      print_error("verify_config()","watch_dir must be defined in GENERAL section")
      return False
  else:
    print_error("verify_config()","GENERAL section must be defined")
    return False
  
  ##TV Section
  if config.has_section('TV'):
    if config.has_option('TV','tv_dir'):
      watch_dir=config.get('TV','tv_dir')
    else:
      print_error("verify_config()","tv_dir must be defined in TV section")
      return False
  else:
    print_error("verify_config()","TV section must be defined")
    return False
  
  return True

def test_classifier(classifier):
  """Test the classifier with some sample training data.
  You can modify the tests by changing the tests variable, each tuple would be a test.
  
  """
  #each tuple is a test
  #[length in seconds]
  tests = ([1200],[2500],[5000],[8000])
  for test in tests:
    print_log("testing "+str(test[0])+" seconds ("+str(test[0]/60)+" minutes)...")
    result = int(classifier.svc.predict(test)[0])
    if result == Video.tv:
      print_log("tv")
    elif result == Video.movie:
      print_log("movie")
    else:
      print_log("no result")
    print_log("...done")

def main():
  
  #Parse command line arguments
  parser = argparse.ArgumentParser("Classify, Rename, and Move media")
  parser.add_argument('--version', action='version', version=version)
  parser.add_argument('-v','--verbose', help="enable verbose output", action='store_true')
  parser.add_argument('--debug', help="enable debug output",action='store_true')
  parser.add_argument('--conf', help="define a configuration file to load", default='mediad.conf')
  parser.add_argument('-d','--daemon', help="start the media daemon", action='store_true')
  parser.add_argument('-p','--plot', help="plot the training data", action='store_true')
  parser.add_argument('-t','--test', help="test the classifier SVM", action='store_true')
  parser.add_argument('-f','--filename', help="classify a specific file", nargs=1)
  global args
  args = parser.parse_args()
  
  #Load config file
  config = ConfigParser.ConfigParser()
  config.read(args.conf)
  
  if verify_config(config) is False:
    print_error_and_exit("Config '%s'" % args.conf,"Error in config file")
  
  if args.verbose:
    print "verbose logging on"
  
  print_log("gathering training data...")
  #1 for tv, 0 for movie
  classifier = Classifier()
  classifier.gather_training_data(config.get("TV","tv_dir"),Video.tv)
  classifier.gather_training_data(config.get("MOVIES","movie_dir"),Video.movie)
  print_log("...done")
  print_log("training SVM...")
  classifier.train()
  print_log("...done")
  if args.plot:
    print_log("plotting training data...")
    classifier.plot_training_data()
    print_log("...done")
  if args.test:
    test_classifier(classifier)
  else:
    if args.filename:
      print_log("classifying file...")
      if os.path.exists(args.filename[0]):
        print_log_verbose("file found: "+str(args.filename[0]))
        result = classifier.classify(svc,args.filename[0])
        if result == Video.tv:
          print_log("tv")
        elif result == Video.movie:
          print_log("movie")
        else:
          print_log("error")
      else:
        print_error("main()","file not found")
      print_log("...done")

if __name__ == "__main__":
  main()

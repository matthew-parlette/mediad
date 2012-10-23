#!/usr/bin/env python
from thetvdb import thetvdb
import argparse
import ConfigParser
import sys
import daemon
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

def get_video_features(filename):
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

def gather_training_set(directory,classification,X = None,y = None):
  """Create a training set of data with the media files existing on the system. For each
  file, gather features to add it to the X array (a num_files X num_features array) and the
  classification in the y vector (a num_files length vector) matching 1 for tv and 0 for movies.
  
  X and y can be passed in if more than one directory is being scanned
  
  This returns a tuple of X and y.
  
  """
  
  for path,subdirs,files in os.walk(directory):
    for filename in files:
      absolute_path = os.path.join(path, filename)
      if args.verbose:
        print_log_verbose("processing file "+filename)
      info = kaa.metadata.parse(absolute_path)
      #only process video files
      #documentation here: http://doc.freevo.org/api/kaa/metadata/usage.html#attributes-keys
      if info is not None and info.media == "MEDIA_AV":
        #gather features
        row = get_video_features(absolute_path)
        #add to training set
        if row is not None:
          if X is None:
            X = array(row)
            y = array([classification])
          else:
            X = vstack((X,row))
            y = hstack((y,[classification]))
  
  return [X,y]

def classify(svc,filename):
  """Classify the given file using the SVM.
  Return a classification from the Video class.
  
  """
  
  features = get_video_features(filename)
  if features is not None:
    print_log_verbose("classifying "+str(filename))
    print_log_verbose("features: "+str(features))
    return int(svc.predict([features])[0])
  else:
    return -1

def test_classifier(svc):
  tests = ([1200],[2500],[5000],[8000])
  for test in tests:
    print_log("testing "+str(test[0])+" seconds ("+str(test[0]/60)+" minutes)...")
    result = int(svc.predict(test)[0])
    if result == Video.tv:
      print_log("tv")
    elif result == Video.movie:
      print_log("movie")
    else:
      print_log("no result")
    print_log("...done")

def plot_training_data(clf,X,Y):
  """Plot the training data to the screen to be used for troubleshooting.
  This was adapted from http://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html
  
  """
  print_log("X:\n"+str(X))
  print_log("y:\n"+str(y))

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
  (X,y) = gather_training_set(config.get("TV","tv_dir"),Video.tv)
  (X,y) = gather_training_set(config.get("MOVIES","movie_dir"),Video.movie,X,y)
  print_log("...done")
  print_log("training SVM...")
  svc = svm.SVC(kernel="linear")
  svc.fit(X,y)
  print_log("...done")
  if args.plot:
    print_log("plotting training data...")
    plot_training_data(svc,X,y)
    print_log("...done")
  if args.test:
    test_classifier(svc)
  else:
    if args.filename:
      print_log("classifying file...")
      if os.path.exists(args.filename[0]):
        print_log_verbose("file found: "+str(args.filename[0]))
        result = classify(svc,args.filename[0])
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

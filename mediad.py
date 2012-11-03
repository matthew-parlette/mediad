#!/usr/bin/env python
from thetvdb import thetvdb
import argparse
import ConfigParser
import sys
import os
import time
import kaa.metadata
import pylab as pl
from numpy import array,vstack,hstack,mgrid,c_
from sklearn import svm
from daemon import Daemon
import pika
import uuid

#Global Arguments
version = '0.1'
args = None
logfile_path = None
config = None
log = None
response = None

#tv = thetvdb.TVShow('94571')

class Video:
  """Easy referencing of video classifications.
  
  """
  tv=1
  movie=0

class Classifier(Daemon):
  status = None
  
  def __init__(self,pidfile,logfile_path = None,amqp_host = 'localhost'):
    #fix the resetting of all variables, variables are shared across instances
    self.svc = svm.SVC(kernel="linear")
    self.__X = None
    self.__y = None
    self.amqp_host = amqp_host
    self.amqp_queue = 'classifyd'
    if args:
      self.log = Logger(logfile_path,args.verbose)
    else:
      self.log = Logger(logfile_path)
    #if not hasattr(self,'status'):
    if Classifier.status is None:
      log.print_log("setting status to initializing")
      Classifier.status = 'initializing'
    #call the parent's __init__ to initialize the daemon variables
    #super(Daemon,self).__init__(pidfile)
    Daemon.__init__(self,pidfile)
    #create the channel
    self.channel = self.setup_channel(delete_if_empty=True)
    if self.channel is None:
      self.log.print_error_and_exit("channel creation failed, classifier exiting...")
  
  def __repr__(self):
    if self.get_pid() is None:
      return "classifyd is not running"
    else:
      return "classifyd is running (status: %s)" % str(Classifier.status)
  
  def get_video_features(self,filename):
    """Gather the features of the given file and return a row to be used in the SVM.
    
    Parameter is the absolute filename
    """
    
    self.log.print_log_verbose("Gathering video features for "+str(filename))
    if os.path.exists(filename):
      info = kaa.metadata.parse(filename)
    else:
      self.log.print_error("file cannot be found")
      return None
    self.log.print_log_verbose("Media type for: "+str(info.media))
    if info is not None and info.media == "MEDIA_AV":
      #gather features
      x1 = int(info.length)
      self.log.print_log_verbose("x1 (video length): "+str(x1))
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
        self.log.print_log_verbose("processing file "+filename)
        info = kaa.metadata.parse(absolute_path)
        #only process video files
        #documentation here: http://doc.freevo.org/api/kaa/metadata/usage.html#attributes-keys
        if info is not None and info.media == "MEDIA_AV":
          #gather features
          row = self.get_video_features(absolute_path)
          #add to training set
          if row is not None:
            self.log.print_log_verbose("adding row: "+str(row))
            X_rows.append(row)
            y_rows.append(classification)
    
    #now add the gathered data to the array
    if len(X_rows) > 0:
      self.log.print_log_verbose("X_rows: "+str(X_rows))
      self.log.print_log_verbose("y_rows: "+str(y_rows))
      if self.__X is None:
        self.__X = array(X_rows)
        self.__y = array(y_rows)
      else:
        self.__X = vstack((self.__X,X_rows))
        self.__y = hstack((self.__y,y_rows))
    

  def train(self):
    self.log.print_log_verbose("training with data\nX:\n%s\ny:\n%s" % (str(self.__X),str(self.__y)))
    Classifier.status = 'training'
    self.svc.fit(self.__X,self.__y)
    time.sleep(30)
    self.log.print_log_verbose("classifier is trained and ready")
    Classifier.status = 'ready'
  
  def classify(self,filename):
    """Classify the given file using the SVM.
    Return a classification from the Video class.
    
    """
    self.log.print_log_verbose("classify(), filename: %s" % str(filename))
    features = self.get_video_features(filename)
    if features is not None:
      self.log.print_log_verbose("classifying "+str(filename))
      self.log.print_log_verbose("features: "+str(features))
      return int(self.svc.predict([features])[0])
    else:
      return -1

  def plot_training_data(self):
    """Plot the training data to the screen to be used for troubleshooting.
    Until it is figured out, this just prints the training data.
    
    """
    self.log.print_log("X:\n"+str(self.__X))
    self.log.print_log("y:\n"+str(self.__y))
  
  def setup_channel(self,delete_if_empty=False):
    """Configure the amqp channel.
    delete_if_empty: delete the queue before declaring it
    Return the created channel object
    
    """
    try:
      #create connection
      connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.amqp_host))
      self.log.print_log_verbose("connection initialized")
      
      #create channel
      channel = connection.channel()
      self.log.print_log("channel initialized")
      
      #if the parameter is passed, then delete the queue, but only if empty
      #if we don't delete the queue and change a setting (durable, for example)
      #  then the queue_declare() bombs
      if delete_if_empty:
        channel.queue_delete(queue=self.amqp_queue, if_empty=True)
      
      #declare the queue
      channel.queue_declare(queue=self.amqp_queue, durable=True)
      self.log.print_log("queue declared")
      
      #qos allows for better handling of multiple clients
      channel.basic_qos(prefetch_count=1)
      return channel
    except:
      #this should be more robust and remove the catch-all except
      self.log.print_error_and_exit("channel not created: %s" % sys.exc_info()[0])
      return None
  
  def run(self):
    """Override for inherited run method of the Daemon class.
    
    """
    while True:
      if Classifier.status == 'initializing':
        self.train()
      elif Classifier.status == 'ready':
        self.log.print_log("classifier daemon is running (pid %s)" % str(os.getpid()))
        if self.channel:
          self.log.print_log_verbose("channel appears available")
          def on_request(ch, method, properties, body):
            self.log.print_log_verbose("received message (delivery tag %s): %s" % (method.delivery_tag,body))
            result = self.classify(body)
            self.log.print_log_verbose("classified as %s" % str(result))
            ch.basic_publish(exchange='',
                             routing_key=properties.reply_to,
                             properties=pika.BasicProperties(correlation_id = properties.correlation_id),
                             body=str(result))
            self.log.print_log_verbose("sent response")
            ch.basic_ack(delivery_tag = method.delivery_tag)
            self.log.print_log_verbose("acknowledged %s" % method.delivery_tag)
          
          #everything is ready to go, now start the consuming of the queue
          self.log.print_log("queue %s declared, listening for messages" % self.amqp_queue)
          self.channel.basic_consume(on_request,queue=self.amqp_queue)
          #the next command blocks, so it will keep listening and this method will no longer loop
          self.channel.start_consuming()
        else:
          self.log.print_error_and_exit("error creating rabbitmq channel")
  
  def stop(self):
    """Override for inherited stop method of Daemon class.
    Right now this just logs that the classifier is stopping.
    
    """
    self.log.print_log("classifier daemon is shutting down (pid %s)" % str(os.getpid()))
    Daemon.stop(self)

class Logger():
  def __init__(self,logfile_path=None,verbose=False):
    if logfile_path is None:
      self.logfile = None
    else:
      self.logfile = open(os.path.abspath(logfile_path),'a+')
    self.verbose = verbose
  
  def __repr__(self):
    if self.logfile is None:
      return "stdout"
    else:
      return self.logfile.name
  
  def close(self):
    if self.logfile is not None:
      logfile.close()
  
  def timestamp(self):
    return time.strftime("%Y-%m-%d %T| ")

  def print_error(self,message):
    """Print the error message with a prefix designating that it is an error.
    This should print to the logfile (if defined) as well as the stdout.
    
    """
    self.print_log_and_stdout("ERROR: %s" % (message))

  def print_error_and_exit(self,message):
    self.print_error(message)
    sys.exit(1)

  def print_log(self,message):
    if self.logfile is None:
      print str(message)
    else:
      message = self.timestamp()+message
      self.logfile.write(message+'\n')
      self.logfile.flush()

  def print_log_and_stdout(self,message):
    self.print_log(message)
    #if logfile is defined, then we already printed it to stdout
    if self.logfile:
      print message
    
  def print_log_verbose(self,message):
    if self.verbose:
      self.print_log(message)

def verify_config(config):
  """Verify that the essential parts of the configuration are provided in the ConfigParser object.
  Return False if an error was found.
  """
  ##General Section
  if config.has_section('GENERAL'):
    if config.has_option('GENERAL','watch_dir'):
      watch_dir=config.get('GENERAL','watch_dir')
    else:
      log.print_error("watch_dir must be defined in GENERAL section")
      return False
  else:
    log.print_error("GENERAL section must be defined")
    return False
  
  ##TV Section
  if config.has_section('TV'):
    if config.has_option('TV','tv_dir'):
      watch_dir=config.get('TV','tv_dir')
    else:
      log.print_error("tv_dir must be defined in TV section")
      return False
  else:
    log.print_error("TV section must be defined")
    return False
  
  return True

def load_media_data(classifier):
  """Load the media data to the passed classifier with the current files from the tv and movies folders.
  Returns the results in a boolean.
  
  """
  log.print_log("gathering training data...")
  classifier.gather_training_data(config.get("TV","tv_dir"),Video.tv)
  classifier.gather_training_data(config.get("MOVIES","movie_dir"),Video.movie)
  log.print_log("...done")
  return True

def test_classifier(classifier):
  """Test the classifier with some sample training data.
  You can modify the tests by changing the tests variable, each tuple would be a test.
  
  """
  #each tuple is a test
  #[length in seconds]
  tests = ([1200],[2500],[5000],[8000])
  for test in tests:
    log.print_log("testing "+str(test[0])+" seconds ("+str(test[0]/60)+" minutes)...")
    result = int(classifier.svc.predict(test)[0])
    if result == Video.tv:
      log.print_log("tv")
    elif result == Video.movie:
      log.print_log("movie")
    else:
      log.print_log("no result")
    log.print_log("...done")

def classify(filename):
  """After verifying the file exists, send the filename to the classifyd message queue and wait for a response.
  
  """
  connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
  channel = connection.channel()
  
  channel.queue_declare(queue='classifyd', durable=True)
  
  #the correlation id will make sure we are reading the response to our request
  corr_id = str(uuid.uuid4())
  
  #setup the response callback
  global response
  response = None
  def on_response(ch, method, props, body):
    log.print_log("received %s" % str(body))
    if corr_id == props.correlation_id:
      global response
      #return int(response.body)
      response = int(body)
  
  #setup the response queue
  result = channel.queue_declare(exclusive=True)
  response_queue = result.method.queue
  channel.basic_consume(on_response, no_ack=True, queue=response_queue)
  
  #delivery_mode=2 means persistent
  #send the request
  channel.basic_publish(exchange='',routing_key='classifyd',
                        properties=pika.BasicProperties(
                                                        reply_to = response_queue,
                                                        correlation_id = corr_id,
                                                        delivery_mode = 2),
                        body=filename)
  log.print_log("sent %s" % str(filename))
  
  #wait for the response
  while response is None:
    connection.process_data_events()
  return int(response)
  
  connection.close()
  return -1

def main():
  
  #Parse command line arguments
  parser = argparse.ArgumentParser("Classify, Rename, and Move media")
  parser.add_argument('--version', action='version', version=version)
  parser.add_argument('-v','--verbose', help="enable verbose output", action='store_true')
  parser.add_argument('--debug', help="enable debug output",action='store_true')
  parser.add_argument('--conf', help="define a configuration file to load", default='mediad.conf')
  parser.add_argument('-d','--daemon', help="manage the media daemon", nargs=1)
  parser.add_argument('-p','--plot', help="plot the training data", action='store_true')
  parser.add_argument('-t','--test', help="test the classifier SVM", action='store_true')
  parser.add_argument('-f','--filename', help="classify a specific file", nargs=1)
  parser.add_argument('--logfile', help="specify a log file for the output", nargs=1)
  parser.add_argument('-c','--classifier', help="manage the classifier daemon", nargs=1)
  global args
  args = parser.parse_args()
  
  #Load config file
  global config
  config = ConfigParser.ConfigParser()
  config.read(args.conf)
  
  global logfile_path
  if args.logfile and args.logfile[0]:
    logfile_path = os.path.abspath(args.logfile[0])
  elif config.has_option("GENERAL","logfile") and len(config.get("GENERAL","logfile")):
    if logfile_path is None:
      logfile_path = os.path.abspath(config.get("GENERAL","logfile"))
  
  global log
  log = Logger(logfile_path)
  
  if verify_config(config) is False:
    log.print_error_and_exit("Error in config file %s" % args.conf)
  
  if args.verbose:
    log.print_log_verbose("verbose logging on")
  
  if args.classifier:
    if not args.classifier[0] or args.classifier[0] not in ('start','stop','restart','status'):
      log.print_error_and_exit("expected classifier argument in {start|stop|restart|status}")
    #at this point, we have a valid daemon command
    classifier = Classifier(config.get("CLASSIFIER","pidfile"),logfile_path)
    if args.classifier[0] in ('start','restart'):
      if args.classifier[0] == 'restart':
        classifier.stop()
      if load_media_data(classifier):
        classifier.start()
        log.print_log_and_stdout(str(classifier)) #why isn't this printing?
      else:
        log.print_error_and_exit("error loading media data")
    elif args.classifier[0] == 'stop':
      classifier.stop()
    elif args.classifier[0] == 'status':
      log.print_log_and_stdout(str(classifier))

  if args.plot:
    log.print_log("plotting training data...")
    
    log.print_log("...done")
  if args.test:
    test_classifier(classifier)
  else:
    if args.filename:
      log.print_log("classifying file...")
      if os.path.exists(args.filename[0]):
        log.print_log_verbose("file found: "+str(args.filename[0]))
        result = classify(args.filename[0])
        if result == Video.tv:
          log.print_log_and_stdout("tv")
        elif result == Video.movie:
          log.print_log_and_stdout("movie")
        else:
          log.print_log_and_stdout("error")
      else:
        log.print_error("file not found")
      log.print_log("...done")

if __name__ == "__main__":
  main()

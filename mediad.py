#!/usr/bin/env python
from thetvdb import thetvdb
import argparse
import ConfigParser
import sys
import os
import time
import kaa.metadata
import pylab as pl
from numpy import array,vstack,hstack,mgrid,c_,shape
from sklearn import svm
from sklearn.externals import joblib
from daemon import Daemon
import pika
import uuid
import traceback
import cPickle as pickle
import StringIO
import re

#Global Arguments
version = '0.5'
args = None
logfile_path = None
config = None
log = None
response = None

config_default="""
[TV]
season_regex=(?ix)(?:s|season|^)\s*(\d{2})
episode_regex=(?ix)(?:e|x|episode|^)\s*(\d{2})
"""

#tv = thetvdb.TVShow('94571')

class Video:
  """Easy referencing of video classifications.
  
  """
  tv=1
  movie=0
  
  @staticmethod
  def to_string(classification):
    if classification == Video.tv:
      return "tv"
    if classification == Video.movie:
      return "movie"
    return str(None)

class Status():
  def __init__(self):
    self.message = 'initializing'
    self.statistics = {}
  
  @property
  def message(self):
    """The status message as a string."""
    return self.message
  
  @message.setter
  def message(self,value):
    self.message = value
  
  def add_stat(self,key,amount=1):
    """Add a number to a particular statistic category.
    By default, if you provide the category (for example 'files') without an amount parameter, then 1 will be added to the 'files' statistic
    
    You can also provide an amount to add to the statistic, including a negative number.
    This should only be used for integer statistics
    """
    if key not in self.statistics:
      self.statistics[key] = amount
    else:
      self.statistics[key] += amount

class Classifier(Daemon):
  
  def __init__(self,pidfile,logfile_path = None,amqp_host = 'localhost',svm_save_filename = None,
               status_filename = None, X_filename = None, y_filename = None):
    if args:
      self.log = Logger(logfile_path,args.verbose)
    else:
      self.log = Logger(logfile_path)
    
    #For progress messages
    self.files_processed = 0
    
    #Make sure pickle supports compress
    pickle.HIGHEST_PROTOCOL
    
    self.svc = svm.SVC(kernel="linear")
    self.X_filename = os.path.abspath(X_filename) if X_filename else None
    self.y_filename = os.path.abspath(y_filename) if y_filename else None
    self.__X = None
    self.__y = None
    self.amqp_host = amqp_host
    self.amqp_queue = 'classifyd'
    self.svm_filename = os.path.abspath(svm_save_filename) if svm_save_filename else None
    
    self.status_filename = status_filename
    if self.status_filename and os.path.exists(self.status_filename):
      #if the file exists, try to load it from there
      f = open(self.status_filename,'rb')
      self.status = pickle.load(f)
      f.close()
    else:
      #if not, then assume we are starting over
      self.status = Status()
    #call the parent's __init__ to initialize the daemon variables
    Daemon.__init__(self,pidfile)
  
  def __repr__(self):
    if self.get_pid() is None:
      return "classifyd is not running"
    else:
      return "classifyd is running (status: %s)" % str(self.status.message)
  
  def get_statistics(self):
    if self.get_pid():
      return str(self.status.statistics)
    return None
  
  def update_status(self,message=None,stat_key=None,stat_value=None):
    if message:
      self.log.print_log_verbose("updating status message to '%s'" % message)
      self.status.message = message
      self.log.print_log_verbose("status message set to '%s'" % message)
    
    if stat_key:
      self.status.add_stat(stat_key,amount=int(stat_value if stat_value else 1))
    
    #if something was updated, save it
    if self.status_filename and (message or stat_key):
      try:
        output = open(self.status_filename,'wb')
        pickle.dump(self.status,output)
        self.log.print_log_verbose("status saved to file '%s'" % self.status_filename)
        output.close()
      except Exception,e:
        self.log.print_error("Error saving status to %s: %s %s" % (str(self.status_filename),sys.exc_info()[0],e))
        self.log.print_error("Traceback: %s" % traceback.format_exc())
  
  def update_progress(self,files_processed=1):
    """Update the user of the progress of the system so far, generally while gathering training data.
    This will update the self.files_processed variable by adding on the files_processed variable.
    At a cutoff, it will update the log of its progress.
    
    """
    self.files_processed += files_processed
    #Update every 500 files
    if self.files_processed % 500 == 0:
      self.log.print_log("Progress update, %s files processed for training data" % str(self.files_processed))
  
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
            self.update_status(stat_key='training examples')
            self.update_progress(files_processed=1)
    
    #now add the gathered data to the array
    if len(X_rows) > 0:
      self.log.print_log_verbose("Adding X_rows: "+str(X_rows))
      self.log.print_log_verbose("Adding y_rows: "+str(y_rows))
      if self.__X is None:
        self.__X = array(X_rows)
        self.__y = array(y_rows)
      else:
        self.__X = vstack((self.__X,X_rows))
        self.__y = hstack((self.__y,y_rows))
  
  def load_pickle(self,filename = None):
    """Load the object from the file and return it.
    
    """
    if filename and os.path.exists(filename):
      try:
        f = open(filename,'rb')
        result = pickle.load(f)
        f.close()
        self.log.print_log_verbose("Loaded object from file %s" % filename)
        self.log.print_log_verbose("Object data: %s" % str(result))
        return result
      except Exception,e:
        self.log.print_error("Pickle count not be loaded from file (%s), error was %s %s" % (str(filename),sys.exc_info()[0],e))
        self.log.print_error("Traceback: %s" % traceback.format_exc())
        return None
    return None
  
  def save_pickle(self,obj,filename = None):
    """Save the object to a file."""
    if filename:
      try:
        output = open(filename,'wb')
        pickle.dump(obj,output)
        output.close()
        self.log.print_log_verbose("save_pickle(): Saved object to file %s" % filename)
        return True
      except Exception,e:
        self.log.print_error("Object could not be saved to file (%s), error was %s %s" % (str(filename),sys.exc_info()[0],e))
        self.log.print_error("Traceback: %s" % traceback.format_exc())
        return False
    return False
  
  def load_svm_from_file(self,filename = None):
    """Load the saved SVM from a file. If the file does not exist or could not be loaded, then return false.
    
    """
    
    if filename and os.path.exists(filename):
      try:
        self.log.print_log("loading SVM from %s..." % filename)
        self.svc = joblib.load(filename)
        self.update_status("ready")
        self.log.print_log("...done")
        
        #X and y should be available for loading as well
        self.__X = self.load_pickle(self.X_filename)
        self.__y = self.load_pickle(self.y_filename)
        
        #Set the status statistic for training examples
        if self.__X is not None and len(self.__X) > 0:
          self.update_status(stat_key='training examples',stat_value=int(len(self.__X)))
        else:
          self.log.print_error("self.__X was empty, but I expected it to have values loaded from a pickle save file. Status statistics will not work for the running daemon")
      except Exception,e:
        self.log.print_error("SVM could not be loaded from file (%s), error was %s %s" % (str(filename),sys.exc_info()[0],e))
        self.log.print_error("Traceback: %s" % traceback.format_exc())
        return False
      return True
    else:
      self.log.print_log_verbose("load_svm_from_file() called with invalid filename (filename: %s)" % str(filename))
      return False

  def train(self):
    """Train the SVM with the current __X matrix and __y vector.
    
    """
    self.log.print_log("training SVM...")
    self.update_status('training')
    #Since we are re-training, we delete the pickled X and y if they exist
    if self.X_filename and os.path.exists(self.X_filename):
      os.remove(self.X_filename)
    if self.y_filename and os.path.exists(self.y_filename):
      os.remove(self.y_filename)
    
    #train with the current __X and __y
    self.svc.fit(self.__X,self.__y)
    self.log.print_log("...done")
    
    #Save the SVM to file
    if self.svm_filename:
      self.log.print_log_verbose("saving SVM to %s" % str(self.svm_filename))
      try:
        joblib.dump(self.svc, self.svm_filename, compress=9)
        self.log.print_log_verbose("SVM saved as %s" % str(self.svm_filename))
      except TypeError:
        #if the compress option is not supported, then we try without
        try:
          joblib.dump(self.svc, self.svm_filename)
          self.log.print_log_verbose("SVM saved as %s (without compression)" % str(self.svm_filename))
        except Exception,e:
          self.log.print_error("Error saving SVM to %s: %s %s" % (str(self.svm_filename),sys.exc_info()[0],e))
      except Exception,e:
        self.log.print_error("Error saving SVM to %s: %s %s" % (str(self.svm_filename),sys.exc_info()[0],e))
        self.log.print_error("Traceback: %s" % traceback.format_exc())
      self.log.print_log_verbose("pickling %s %s" % (str(self.__X),self.X_filename))
      
      #Save the X and y variables
      if self.save_pickle(self.__X,self.X_filename):
        self.log.print_log("X matrix (size %s) saved to %s" % (shape(self.__X),self.X_filename))
        if self.save_pickle(self.__y,self.y_filename):
          self.log.print_log("y vector (size %s) saved to %s" % (shape(self.__y),self.y_filename))
        else:
          self.log.print_log_error("Error saving y vector pickle")
      else:
        self.log.print_log_error("Error saving X Matrix pickle")
        
    self.log.print_log_verbose("returning from train(): classifier is trained and ready")
    self.update_status('ready')
  
  def classify(self,filename):
    """Classify the given file using the SVM.
    Return a classification from the Video class.
    
    """
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
    
    This method is not working and should be avoided for now.
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
      #channel.queue_delete(queue=self.amqp_queue, if_empty=True)
      if delete_if_empty:
        #since there is no method for queue_exists, we use a try block
        try:
          channel.queue_delete(queue=self.amqp_queue, if_empty=True)
        except pika.exceptions.AMQPChannelError, e:
          self.log.print_log_verbose("tried to delete %s queue but received an error. If this is 404, it should be no problem. Error was %s" % (self.amqp_queue,str(e)))
      
      #declare the queue
      self.log.print_log("declaring queue")
      channel.queue_declare(queue=self.amqp_queue, durable=True)
      self.log.print_log("queue declared")
      
      #qos allows for better handling of multiple clients
      channel.basic_qos(prefetch_count=1)
      return channel
    except Exception,e:
      #this should be more robust and remove the catch-all except
      self.log.print_error_and_exit("channel not created: %s (%s)" % (str(e),sys.exc_info()[0]))
      return None
  
  def run(self):
    """Override for inherited run method of the Daemon class.
    
    """
    self.log.print_log_verbose("run() called. classifier status is %s." % str(self.status.message))
    
    while True:
      if self.status.message == 'initializing':
        self.train()
      elif self.status.message == 'ready':
        self.log.print_log("classifier daemon is running (pid %s)" % str(os.getpid()))
        #channel = self.setup_channel(delete_if_empty=True)
        
        """
        This code below is copied from setup_channel() method because the queue_declare call
        seems to block (when I expect the start_consuming() to be blocking instead), so I'll keep the
        copied code below until I figure out how to get the channel from another method.
        """
        
        ###################################
        #this is copied from setup_channel#
        ###################################
        
        try:
          #create connection
          connection = pika.BlockingConnection(pika.ConnectionParameters(host=self.amqp_host))
          self.log.print_log_verbose("connection initialized")
          
          #create channel
          channel = connection.channel()
          self.log.print_log_verbose("channel initialized")
          
          #declare the queue
          self.log.print_log_verbose("delcaring queue")
          channel.queue_declare(queue=self.amqp_queue, durable=True, exclusive=False, auto_delete=False)
          self.log.print_log_verbose("queue declared")
          
          #qos allows for better handling of multiple clients
          channel.basic_qos(prefetch_count=1)
        except Exception,e:
          #this should be more robust and remove the catch-all except
          self.log.print_error_and_exit("channel not created: %s (%s)" % (str(e),sys.exc_info()[0]))
        
        ###################################
        #this is copied from setup_channel#
        ###################################
        
        if channel:
          self.log.print_log_verbose("channel appears available")
          def on_request(ch, method, properties, body):
            self.log.print_log("received message (delivery tag %s): %s" % (method.delivery_tag,body))
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
          self.log.print_log("queue %s declared, listening for messages..." % self.amqp_queue)
          channel.basic_consume(on_request,queue=self.amqp_queue)
          #the next command blocks, so it will keep listening and this method will no longer loop
          channel.start_consuming()
        else:
          self.log.print_error_and_exit("rabbitmq channel creation failed, classifier exiting...")
  
  def stop(self):
    """Override for inherited stop method of Daemon class.
    Right now this just logs that the classifier is stopping.
    
    """
    self.log.print_log("classifier daemon is shutting down (pid %s)" % str(os.getpid()))
    if self.status_filename and os.path.exists(self.status_filename):
      os.remove(self.status_filename)
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

class MediaFile():
  def __init__(self,original_filename,classification = None,db_search_term = None):
    self.original_path = os.path.dirname(original_filename)
    self.original_filename = os.path.basename(original_filename)
    self.classification = classification
    self.db_search_term = db_search_term
    self.db_search_results = None
    self.db_object = None
    self.new_filename = None
    self.new_path = None
    self.exception = False
  
  def __repr__(self):
    s = "Original Filename: %s\n" % str(self.original_abspath())
    s += "Classification: %s\n" % str(Video.to_string(self.classification))
    s += "DB Search Term: %s\n" % str(self.db_search_term)
    s += "DB Search Results: %s\n" % str(self.db_search_results)
    s += "DB Object: %s\n" % str(self.db_object)
    s += "New Filename: %s\n" % str(self.new_abspath())
    s += "Exception: %s" % str(self.exception)
    return s
  
  def __eq__(self, other):
    if hasattr(other,'original_abspath') and callable(other.original_abspath):
      return os.path.samefile(self.original_abspath(),other.original_abspath())
    return False
  
  def __ne__(self, other):
    if hasattr(other,'original_abspath') and callable(other.original_abspath):
      return not os.path.samefile(self.original_abspath(),other.original_abspath())
    return True
  
  def original_abspath(self):
    """The absolute path of the original file."""
    return os.path.join(self.original_path,self.original_filename) if self.original_path and self.original_filename else None
  
  def new_abspath(self):
    """The absolute path of the new (renamed) file."""
    return os.path.join(self.new_path,self.new_filename) if self.new_path and self.new_filename else None
  
  def is_exception(self):
    return self.exception
  
  def classify(self):
    """Classify the original file and save it as the instance classification variable.
    
    This will also return the classification as 0 or 1."""
    
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    
    log.print_log_verbose("declaring queue")
    channel.queue_declare(queue='classifyd', durable=True)
    log.print_log_verbose("queue declared")
    
    #the correlation id will make sure we are reading the response to our request
    self.corr_id = str(uuid.uuid4())
    
    #setup the response callback
    self.classification = None
    def on_response(ch, method, props, body):
      log.print_log_verbose("received response %s (%s)" % (str(body),str(Video.to_string(body))))
      if self.corr_id == props.correlation_id:
        self.classification = int(body)
    
    #setup the response queue
    response_queue = channel.queue_declare(exclusive=True).method.queue
    channel.basic_consume(on_response, no_ack=True, queue=response_queue)
    
    #send the request
    channel.basic_publish(exchange='',routing_key='classifyd',
                          properties=pika.BasicProperties(
                                                          reply_to = response_queue,
                                                          correlation_id = self.corr_id,
                                                          delivery_mode = 2),
                          body=self.original_abspath())
    log.print_log_verbose("sent %s" % str(self.original_abspath()))
    
    #wait for the response
    while self.classification is None:
      connection.process_data_events()
    
    connection.close()
    return self.classification
  
  def search(self):
    """Search the media database (either tv or movie) for the original filename."""
    if self.db_search_term is None:
      #set the search term equal to the original filename
      self.db_search_term = self.original_filename
    
    self.db_search_results = {}
    
    if self.classification is Video.tv:
      tvdb = thetvdb.TVShow()
      
      """Next we will try to search for the show. We break on the period, as that is most
      common in the files to separate the show title from the flags and so on.
      When we run out of the search term to trim, then we give up."""
      
      while self.db_search_results == {} and self.exception is False:
        log.print_log_verbose("searching thetvdb for '%s'" % self.db_search_term)
        self.db_search_results = tvdb.search(self.db_search_term)
        if self.db_search_results == {}:
          if '.' in self.db_search_term:
            try:
              self.db_search_term = self.db_search_term[:self.db_search_term.rindex('.')]
              log.print_log_verbose("no results for previous search term, shortening to %s" % self.db_search_term)
            except ValueError:
              self.exception = True #just move on if the period couldn't be found
          else:
            log.print_error("All search terms failed, giving up on this file")
            self.exception = True #give up
    
    if self.classification is Video.movie:
      pass
    
    return self.db_search_results
  
  def select_object_from_search_results(self,db_id = None):
    """Using this instance's search results, pick the object to use when renaming this media file.
    
    For example, select a tv show from the search results to use when renaming an episode.
    
    db_id: The id of the show or movie in its database
    """
    
    if db_id:
      """If a database id was passed in, then set the db_object based on that."""
      if db_id in self.db_search_results:
        if self.classification == Video.tv:
          self.db_object = thetvdb.TVShow(db_id)
          return True
        if self.classification == Video.movie:
          pass
      else:
        log.print_error("The id passed to MediaFile.select_object_from_search_results() was not in the search results")
    else:
      """If the database id is not passed in, then we try to select one from the search results."""
      if len(self.db_search_results) == 1:
        self.db_object = thetvdb.TVShow(self.db_search_results.keys()[0])
        return True
      else:
        log.print_error("There is more than one entry in the search results, I cannot decide which one to use. Marking this file as an exception")
    
    return False
  
  def set_new_path(self,base_path,season_number = None):
    """Set the new_path instance variable from the db_object."""
    if self.db_object:
      if self.classification == Video.tv:
        self.new_path = os.path.join(base_path,self.db_object.get_samba_show_name(),"Season %s" % str(int(season_number)))
        log.print_log_verbose("set new_path to %s" % self.new_path)
        return True
      if self.classification == Video.movie:
        pass
    return False
  
  def set_new_filename(self):
    """Set the new_filename instance variables with the information in the db_object.
    
    This also calls the set_new_path method to populate self.new_path."""
    
    if self.db_object:
      if self.classification == Video.tv:
        """Here we first try to get the season and episode number from the original filename.
        
        This was copied by the very helpful answer from unutbu http://stackoverflow.com/a/9129707
        
        match = re.search(
          r'''(?ix)                 # Ignore case (i), and use verbose regex (x)
          (?:                       # non-grouping pattern
            e|x|episode|^           # e or x or episode or start of a line
            )                       # end non-grouping pattern 
          \s*                       # 0-or-more whitespaces
          (\d{2})                   # exactly 2 digits
          ''', filename)
        """
        season_regex = config.get("TV","season_regex")
        episode_regex = config.get("TV","episode_regex")
        
        log.print_log_verbose("season_regex for tv show: %s" % str(season_regex))
        season_match = re.search(season_regex, self.original_filename)
        if season_match:
          log.print_log_verbose("season_regex matched %r" % season_match.groups())
          season = season_match.groups()[0]
          
          self.set_new_path(config.get("TV","tv_dir"),season_number = int(season))
        
          log.print_log_verbose("episode_regex for tv show: %s" % str(episode_regex))
          episode_match = re.search(episode_regex, self.original_filename)
          if episode_match:
            log.print_log_verbose("episode_regex matched %r" % episode_match.groups())
            episode = episode_match.groups()[0]
            if self.db_object:
              self.new_filename = self.db_object.get_samba_filename(season,episode)
              if self.db_object.error_message:
                print self.db_object.error_message
              log.print_log_verbose("new filename set to %s" % self.new_filename)
              return True
      if self.classification == Video.movie:
        return False
    else:
      log.print_error("set_new_filename called, but db_object is none")
    return False
  
  def process(self,move_file = True):
    """Rename and move the file.
    
    move_file: Move the file if everything looks good (search returned one result)
    
    Return True if everything went as expected and the file was moved.
    Return False if there was any problem with the process. This will also set the exception flag.
    
    """
    
    if os.path.exists(self.original_abspath()):
      if self.classification is None:
        self.classify()
        #print "\n========== after classify ==========\n%s" % self
      if self.classification is not None:
        if self.search():
          #print "\n========== after search ==========\n%s" % self
          if self.select_object_from_search_results():
            #print "\n========== after select object ==========\n%s" % self
            if self.set_new_filename():
              #print "\n========== after set filename ==========\n%s" % self
              return True
            else:
              log.print_error("Set filename failed")
          else:
            log.print_error("Couldn't select one object from the search results")
        else:
          log.print_error("Search failed")
      else:
        log.print_error("Classification failed")
    else:
      log.print_error("Original file could not be found")
    return False

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
  if config.has_option("CLASSIFIER","svm_filename") and classifier.load_svm_from_file(config.get("CLASSIFIER","svm_filename")):
    log.print_log_verbose("SVM loaded from file (%s)" % config.get("CLASSIFIER","svm_filename"))
    return True
  else:
    log.print_log_verbose("svm_filename not provided or load failed. Now loading from scratch")
  
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
  
  log.print_log_verbose("declaring queue")
  channel.queue_declare(queue='classifyd', durable=True)
  log.print_log_verbose("queue declared")
  
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

def add_exception(exceptions,mediafile,save_filename = None):
  """Add the parameter MediaFile object to the exceptions list and save it to a file."""
  
  if mediafile not in exceptions:
    exceptions.append(mediafile)
    exceptions.sort(key=lambda mf: mf.original_filename)
    
    if save_filename:
      try:
        pickle.dump(mediafile,open(save_filename,'wb'))
      except pickle.PicklingError:
        log.print_error("Could not pickle the exceptions list to %s" % save_filename)
      except IOError:
        log.print_error("File could not be opened for writing: %s" % save_filename)

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
  parser.add_argument('-e','--exceptions', help="handle media file exceptions (ex. files that could not be classified)", action='store_true')
  global args
  args = parser.parse_args()
  
  #Load config file
  """First we set the defaults for the config. We read the defaults first, then read the
  actual config file that overwrites any settings are are defined by the user.
  
  This method is copied from: http://bytes.com/topic/python/answers/462831-default-section-values-configparser
  """
  
  default_cfg = StringIO.StringIO(config_default)
  
  global config
  config = ConfigParser.ConfigParser()
  config.readfp(default_cfg)
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
    svm_filename = config.get("CLASSIFIER","svm_filename") if config.has_option("CLASSIFIER","svm_filename") else None
    status_filename = config.get("CLASSIFIER","status_filename") if config.has_option("CLASSIFIER","status_filename") else None
    X_filename = config.get("CLASSIFIER","X_filename") if config.has_option("CLASSIFIER","X_filename") else None
    y_filename = config.get("CLASSIFIER","y_filename") if config.has_option("CLASSIFIER","y_filename") else None
    
    classifier = Classifier(config.get("CLASSIFIER","pidfile"),logfile_path,svm_save_filename=svm_filename,
                                       status_filename=status_filename,X_filename=X_filename,y_filename=y_filename)
    if args.classifier[0] in ('start','restart'):
      if args.classifier[0] == 'restart':
        classifier.stop()
      if classifier.get_pid():
        #if the classifier is already running, then we won't load the media data again
        log.print_log_and_stdout("classifier daemon already running (pid %s)" % classifier.get_pid())
      else:
        if load_media_data(classifier):
          classifier.start()
          log.print_log_and_stdout(str(classifier)) #why isn't this printing?
        else:
          log.print_error_and_exit("error loading media data")
    elif args.classifier[0] == 'stop':
      classifier.stop()
    elif args.classifier[0] == 'status':
      log.print_log_and_stdout(str(classifier))
      log.print_log_and_stdout("statistics: %s" % str(classifier.get_statistics()))

  #Load the exceptions if the file exists
  exceptions = []
  if config.has_option("GENERAL","exceptions_filename"):
    try:
      exceptions = pickle.load(open(config.get("GENERAL","exceptions_filename"),'rb'))
    except pickle.UnpicklingError:
      log.print_error("Error loading exceptions list from %s" % config.get("GENERAL","exceptions_filename"))
    except IOError:
      log.print_error("%s not found, using a blank exceptions list" % config.get("GENERAL","exceptions_filename"))

  if args.plot:
    log.print_log("plotting training data...")
    
    log.print_log("...done")
  if args.test:
    test_classifier(classifier)
  
  if args.filename:
    f = MediaFile(args.filename[0])
    log.print_log_and_stdout("\n\nprocess results: %s\n\n========== final MediaFile instance ==========\n%s" % (str(f.process()),str(f)))
    if f.is_exception():
      add_exception(exceptions,f,save_filename = config.get("GENERAL","exceptions_filename"))
  
  if args.exceptions:
    """For the exceptions client, we display a menu to select the media file to process,
    then allow the user to change any setting in the media file object before processing it."""
    cmd = ''
    while cmd != 'q':
      print "\n\n"
      print "Exceptions Menu"
      print "===============\n"
      print "(L)ist Exceptions"
      print "(Q)uit"
      print "\nExceptions: %s" % str(len(exceptions))
      cmd = raw_input('> ').lower()

if __name__ == "__main__":
  main()

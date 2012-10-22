#!/usr/bin/env python
from thetvdb import thetvdb
import argparse
import ConfigParser
import sys
import daemon

#Global Arguments
version = '0.1'

#tv = thetvdb.TVShow('94571')

def print_error(section,message):
  print "ERROR: %s: %s" % (section,message)

def print_error_and_exit(section,message):
  print_error(section,message)
  sys.exit(1)

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
      print_error("verify_config()","WATCH_DIR must be defined in GENERAL section")
      return False
  else:
    print_error("verify_config()","GENERAL section must be defined")
    return False
  
  ##TV Section
  if config.has_section('TV'):
    if config.has_option('TV','tv_dir'):
      watch_dir=config.get('TV','tv_dir')
    else:
      print_error("verify_config()","TV_DIR must be defined in TV section")
      return False
  else:
    print_error("verify_config()","TV section must be defined")
    return False
  
  return True

def main():
  
  #Parse command line arguments
  parser = argparse.ArgumentParser("Classify, Rename, and Move media")
  parser.add_argument('--version', action='version', version=version)
  parser.add_argument('-v','--verbose', help="enable verbose output", action='store_true')
  parser.add_argument('--debug', help="enable debug output",action='store_true')
  parser.add_argument('--conf', help="define a configuration file to load", default='mediad.conf')
  parser.add_argument('-d','--daemon', help="start the media daemon", action='store_true')
  args = parser.parse_args()
  
  #Load config file
  config = ConfigParser.ConfigParser()
  config.read(args.conf)
  
  if verify_config(config) is False:
    print_error_and_exit("Config '%s'" % args.conf,"Error in config file")
  
  print config.get("GENERAL","video_ext").split(',')

if __name__ == "__main__":
  main()

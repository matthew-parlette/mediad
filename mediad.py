#!/usr/bin/env python
from thetvdb import thetvdb
import argparse
import ConfigParser
import sys

#Global Arguments
version = '0.1'

#tv = thetvdb.TVShow('94571')

def print_error(section,message):
  print "ERROR: %s: %s" % (section,message)

def print_error_and_exit(section,message):
  print_error(section,message)
  sys.exit(1)

def main():
  
  #Parse command line arguments
  parser = argparse.ArgumentParser("Classify, Rename, and Move media")
  #http://docs.python.org/library/argparse.html#adding-arguments
  parser.add_argument('--version', action='version', version=version)
  parser.add_argument('-v','--verbose', help="enable verbose output", action='store_true')
  parser.add_argument('--debug', help="enable debug output",action='store_true')
  parser.add_argument('--conf', help="define a configuration file to load", default='mediad.conf')
  parser.add_argument('-d','--daemon', help="start the media daemon", action='store_true')
  args = parser.parse_args()
  
  #Load config file
  config = ConfigParser.RawConfigParser()
  config.read(args.conf)
  #General Section
  if config.has_section('GENERAL'):
    if config.has_option('GENERAL','watch_dir'):
      watch_dir=config.get('GENERAL','watch_dir')
    else:
      print_error_and_exit("Config '%s'" % args.conf,"WATCH_DIR must be defined in GENERAL section")
  else:
    print_error_and_exit("Config '%s'" % args.conf,"GENERAL section must be defined")
  
  #TV Section
  if config.has_section('TV'):
    if config.has_option('TV','tv_dir'):
      watch_dir=config.get('TV','tv_dir')
    else:
      print_error_and_exit("Config '%s'" % args.conf,"TV_DIR must be defined in TV section")
  else:
    print_error_and_exit("Config '%s'" % args.conf,"TV section must be defined")
  
  #Do Something

if __name__ == "__main__":
  main()

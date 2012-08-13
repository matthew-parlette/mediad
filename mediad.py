#!/usr/bin/env python
from thetvdb import thetvdb
import argparse
import ConfigParser

#Global Arguments
version = '0.1'
config_file = 'mediad.conf'

tv = thetvdb.TVShow('94571')

def main():
  
  #Parse command line arguments
  parser = argparse.ArgumentParser("Classify, Rename, and Move media")
  #http://docs.python.org/library/argparse.html#adding-arguments
  #parser.add_argument('--version', action='version', version=version)
  parser.add_argument('--conf', help="Define a configuration file to load")
  options,arguments = parser.parse_args()
  
  #Load config options
  config = ConfigParser.RawConfigParser()
  config.read(config_file)

if __name__ == "__main__":
  main()

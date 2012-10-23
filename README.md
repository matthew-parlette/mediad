mediad
======

Media file processing daemon to automatically standardize file names and move to a shared media location

Prerequisites
=============

Python packages:
apt-get install python-sklearn python-kaa-metadata python-daemon

Optional (for all features):
apt-get install python-matplotlib

Installation
============

git clone --recursive https://github.com/matthew-parlette/mediad.git

Make sure the recursive option is there to also get the submodules. If, for some reason, the submodules don't load (for example, thetvdb directory is empty), you can run this inside the mediad directory that was created from the git clone command:

git submodule update --init

Modify the mediad.conf to set your tv and movie directories

Usage
=====

Further description to come, but some commands:

See all available commands:
mediad.py --help

Test a few video lengths to make sure the SVM is working:
mediad.py --test

Classify a specific file:
mediad.py --filename /home/matt/video.avi

Citations
=========

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

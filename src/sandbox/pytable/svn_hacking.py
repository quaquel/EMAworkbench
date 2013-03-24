'''
Created on Apr 26, 2012

@author: localadmin
@version: $Revision: 811 $

'''
a = '$Revision: 811 $'
SVN_ID = '$Id: svn_hacking.py 811 2012-04-26 13:36:03Z jhkwakkel $'

# use the built in regular expression library
import re
 
# the subversion Id keyword
svnid = '$Id: svn_hacking.py 811 2012-04-26 13:36:03Z jhkwakkel $'
 
# bow for the mighty regular expression
svnidrep = r'^\$Id: (?P<filename>.+) (?P<revision>\d+) (?P<date>\d{4}-\d{2}-\d{1,2}) (?P<time>\d{2}:\d{2}:\d{2})Z (?P<user>\w+) \$$'
 
# parse the svn Id
mo = re.match(svnidrep, svnid)
 
# use it, like for example:
print 'this is %s - revision %s (%s)' % mo.group('filename', 'revision', 'date')
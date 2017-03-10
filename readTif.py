# -*- coding: utf-8 -*-
"""
Created on Sat Jul 02 11:29:49 2016

@author: andy
"""
from struct import *
import numpy as np
import matplotlib.pyplot as plt

class tifFile(object):
    def __init__ (self, filename):
        ###init blank variables###
    
        self.filename = filename
        self.img = open(self.filename, 'rb')
        self.getHdr()
        self.img.seek(self.IFDOffset)
        self.numTags, = unpack('{0}H'.format(self.byteOrder), self.img.read(2))
        self.getTags()
        #print self.__str__() #print what we have
        self.getStripOffsets()
        #self.image = np.zeros([self.height,self.width]) #will read as np.float64
        self.image = np.zeros([self.height,self.width], dtype=np.uint16) #will read as np.uint16
        self.readImage()
        self.img.close()
        
    def getHdr(self):
        self.img.seek(0)
        #get the identifier
        #if 'II' intel byte-order, that is little endian lowest byte stored first
        #'MM' is motorola byte-order, big-endian highest byte stored first
        #first character of the format string can be used to indicate the byte order
        # '<' little-endian, '>' big-endian
        
        self.Identifier = unpack('BB',self.img.read(2))
        if self.Identifier[0] == 73:
            print 'Intel Byte-order Mark'
            self.byteOrder = '<'
        elif self.Identifier[0] == 77:
            print 'Motorola Byte-order Mark'
            self.byteOrder = '>'
        else:
            raise Exception('Unknown Byte-order Mark')
        
        self.Version, self.IFDOffset  = unpack('{0}HL'.format(self.byteOrder), self.img.read(6))
        #print self.Version, self.IFDOffset
        return
        
    def getTags(self):
        """
        WORD   TagId;       The tag identifier
        WORD   DataType;    The scalar type of the data items
        DWORD  DataCount;   The number of items in the tag data
        DWORD  DataOffset;  The byte offset to the data items
        """
        
#        self.width
#        self.height
#        self.bitDepth
#        self.compression
#        self.stripOffset
#        self.rowsPerStrip
#        self.stripByteCounts
#        self.xRes
#        self.yRes
#        self.ResUnits
        
        for x in range(self.numTags):
            atag = unpack('{0}HHLL'.format(self.byteOrder), self.img.read(12))
            #print atag
            if atag[0] == 256:
                self.width = atag[3]
            elif atag[0] == 257:
                self.height = atag[3]
            elif atag[0] == 258:
                self.bitDepth = atag[3]
            elif atag[0] == 259:
                if atag[3] == 1:
                    self.compression = (1, 'None')
                else:
                    print 'Unknown Compression'
                    
            #strip tags
            elif atag[0] == 273:
                #print atag
                self.stripOffset = atag
                
            #the number of rows of compressed bitmapped data found in each strip
            #Default is  2**32 -  1, which  is effectively infinity.  
            #That is, the entire  image is  one strip.
            elif atag[0] == 278:
                self.rowsPerStrip = atag[3]
                
            elif atag[0] == 279:
                self.stripByteCounts = atag[3]
            elif atag[0] == 282:
                self.xRes = atag[3] #pixels per resolution unit
            elif atag[0] == 283:
                self.yRes = atag[3]
            elif atag[0] == 296:
                if atag[3] == 2:
                    self.ResUnits = (2, 'Inch')
            else:
                #print('Unread tag: {0} contains {1}'.format(atag[0],atag[3]))
                pass
                #277  samples per pixel
        return
        
    def getStripOffsets(self):
#        if self.stripOffset[1] == 4:
#            self.stripOffsetDataType = 'L' #LONG
    
        self.stripOffsets = []
        self.img.seek(self.stripOffset[3])
        
        for aStrip in range(self.stripOffset[2]):
            a, = unpack('{0}L'.format(self.byteOrder),self.img.read(4))
            self.stripOffsets.append(a)
        
        #there are this many strip offsets
        #print 'Number of strip offsets: {0}'.format(len(self.stripOffsets))
        #if there is only one strip offset, then the stripOffset tag contains the
        #index of the first data byte
        if len(self.stripOffsets) == 1:
            #print self.stripOffsets
            self.stripOffsets = []
            self.stripOffsets.append(self.stripOffset[3])
            
        return
        
    def readImage(self):
        #for row in range(self.height):
#        for row in range(2):
#            for col in range(self.width):
#                #try just reading from the first strip offset:
#                self.img.seek(self.stripOffsets[0])
#                a, = unpack('H',self.img.read(2))
#                self.image[col][row] = a
#                print(col, row, a)
        
        self.img.seek(self.stripOffsets[0])
        #print self.img.tell()
        try:
            for x in np.nditer(self.image, op_flags=['readwrite']):
                a, = unpack('{0}H'.format(self.byteOrder),self.img.read(2))
                x[...] = a
        except Exception as e:
            print e
            print self.img.tell()
            
        
        #f, ax = plt.subplots()
        #ax.imshow(self.image)
        return
        
    def __str__(self):
        aStr = 'Header\nFormat: {0}, Version: {1}, IFDOffset: {2}\n'\
            .format(self.Identifier, self.Version, self.IFDOffset)
        aStr += 'Tags\nWidth: {0}, Height: {1}, BitDepth: {2}\n'\
            .format(self.width,self.height,self.bitDepth)
        try:
            aStr += 'Compression: {0}\n'\
                .format(self.compression)
        except:
            pass
        aStr += 'StripOffset: {0}, RowsPerStrip: {1}, StripByteCounts: {2}\n'\
            .format(self.stripOffset,self.rowsPerStrip,self.stripByteCounts)
        try:
            aStr += 'XRes: {0}, YRes: {1}, Units: {2}'\
                .format(self.xRes, self.yRes, self.ResUnits)
        except:
            pass
        return aStr

if __name__ == '__main__':
    #aTif = tifFile('20160628-coPARN.S87A(p1).Mg.tif')
    aTif = tifFile('20160705-coPARN-N470A(p1)Mg2.tif')
    print aTif
    #print aTif.height, aTif.width
    plt.imshow(aTif.image)


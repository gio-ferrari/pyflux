# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:54:08 2018

@author: Luciano A. Masullo
"""

import time
import sys
import struct
import numpy as np
import numba

# Tag Types
tyEmpty8      = struct.unpack(">i", bytes.fromhex("FFFF0008"))[0]
tyBool8       = struct.unpack(">i", bytes.fromhex("00000008"))[0]
tyInt8        = struct.unpack(">i", bytes.fromhex("10000008"))[0]
tyBitSet64    = struct.unpack(">i", bytes.fromhex("11000008"))[0]
tyColor8      = struct.unpack(">i", bytes.fromhex("12000008"))[0]
tyFloat8      = struct.unpack(">i", bytes.fromhex("20000008"))[0]
tyTDateTime   = struct.unpack(">i", bytes.fromhex("21000008"))[0]
tyFloat8Array = struct.unpack(">i", bytes.fromhex("2001FFFF"))[0]
tyAnsiString  = struct.unpack(">i", bytes.fromhex("4001FFFF"))[0]
tyWideString  = struct.unpack(">i", bytes.fromhex("4002FFFF"))[0]
tyBinaryBlob  = struct.unpack(">i", bytes.fromhex("FFFFFFFF"))[0]

# Record types
rtPicoHarpT3     = struct.unpack(">i", bytes.fromhex('00010303'))[0]
rtPicoHarpT2     = struct.unpack(">i", bytes.fromhex('00010203'))[0]
rtHydraHarpT3    = struct.unpack(">i", bytes.fromhex('00010304'))[0]
rtHydraHarpT2    = struct.unpack(">i", bytes.fromhex('00010204'))[0]
rtHydraHarp2T3   = struct.unpack(">i", bytes.fromhex('01010304'))[0]
rtHydraHarp2T2   = struct.unpack(">i", bytes.fromhex('01010204'))[0]
rtTimeHarp260NT3 = struct.unpack(">i", bytes.fromhex('00010305'))[0]
rtTimeHarp260NT2 = struct.unpack(">i", bytes.fromhex('00010205'))[0]
rtTimeHarp260PT3 = struct.unpack(">i", bytes.fromhex('00010306'))[0]
rtTimeHarp260PT2 = struct.unpack(">i", bytes.fromhex('00010206'))[0]
rtMultiHarpNT3   = struct.unpack(">i", bytes.fromhex('00010307'))[0]
rtMultiHarpNT2   = struct.unpack(">i", bytes.fromhex('00010207'))[0]

def readHeaders(inputfile):
    
    magic = inputfile.read(8).decode("utf-8").strip('\0')
    if magic != "PQTTTR":
        print("ERROR: Magic invalid, this is not a PTU file.")
        inputfile.close()
        exit(0)
    
    version = inputfile.read(8).decode("utf-8").strip('\0')
    print('Version', version)

    tagDataList = []    # Contains tuples of (tagName, tagValue)
    while True:
        tagIdent = inputfile.read(32).decode("utf-8").strip('\0')
        tagIdx = struct.unpack("<i", inputfile.read(4))[0]
        tagTyp = struct.unpack("<i", inputfile.read(4))[0]
#        print(tagTyp)
        if tagIdx > -1:
            evalName = tagIdent + '(' + str(tagIdx) + ')'
        else:
            evalName = tagIdent
#        outputfile.write("\n%-40s" % evalName)
        if tagTyp == tyEmpty8:
            inputfile.read(8)
#            outputfile.write("<empty Tag>")
            tagDataList.append((evalName, "<empty Tag>"))
        elif tagTyp == tyBool8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            if tagInt == 0:
#                outputfile.write("False")
                tagDataList.append((evalName, "False"))
            else:
#                outputfile.write("True")
                tagDataList.append((evalName, "True"))
        elif tagTyp == tyInt8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
#            outputfile.write("%d" % tagInt)
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyBitSet64:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
#            outputfile.write("{0:#0{1}x}".format(tagInt,18))
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyColor8:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
#            outputfile.write("{0:#0{1}x}".format(tagInt,18))
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyFloat8:
            tagFloat = struct.unpack("<d", inputfile.read(8))[0]
#            outputfile.write("%-3E" % tagFloat)
            tagDataList.append((evalName, tagFloat))
        elif tagTyp == tyFloat8Array:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
#            outputfile.write("<Float array with %d entries>" % tagInt/8)
            tagDataList.append((evalName, tagInt))
        elif tagTyp == tyTDateTime:
            tagFloat = struct.unpack("<d", inputfile.read(8))[0]
            tagTime = int((tagFloat - 25569) * 86400)
            print('tagTime', tagTime)
            tagTime = time.gmtime(tagTime)
#            outputfile.write(time.strftime("%a %b %d %H:%M:%S %Y", tagTime))
            tagDataList.append((evalName, tagTime))
        elif tagTyp == tyAnsiString:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagString = inputfile.read(tagInt).decode("utf-8").strip("\0")
#            outputfile.write("%s" % tagString)
            tagDataList.append((evalName, tagString))
        elif tagTyp == tyWideString:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
            tagString = inputfile.read(tagInt).decode("utf-16le", errors="ignore").strip("\0")
#            outputfile.write(tagString)
            tagDataList.append((evalName, tagString))
        elif tagTyp == tyBinaryBlob:
            tagInt = struct.unpack("<q", inputfile.read(8))[0]
#            outputfile.write("<Binary blob with %d bytes>" % tagInt)
            tagDataList.append((evalName, tagInt))
        else:
            print("ERROR: Unknown tag type")
            exit(0)
        if tagIdent == "Header_End":
            break
    
    # Reformat the saved data for easier access
    tagNames = [tagDataList[i][0] for i in range(0, len(tagDataList))]
    tagValues = [tagDataList[i][1] for i in range(0, len(tagDataList))]
    
    # get important variables from headers
    numRecords = tagValues[tagNames.index("TTResult_NumberOfRecords")]
    globRes = tagValues[tagNames.index("MeasDesc_GlobalResolution")]
    timeRes = tagValues[tagNames.index("MeasDesc_Resolution")]

    return numRecords, globRes, timeRes   

def readPT3(inputfile, numRecords):
    
    oflcorrection = 0
    dlen = 0
    T3WRAPAROUND = 65536
    
    dtime_array = np.zeros(numRecords)
    truensync_array = np.zeros(numRecords)
    t0 = time.time()
    for recNum in range(0, numRecords):

        try:
            recordData = "{0:0{1}b}".format(struct.unpack("<I", inputfile.read(4))[0], 32)
        except:
            print("The file ended earlier than expected, at record %d/%d."\
                  % (recNum, numRecords))
            exit(0)

        channel = int(recordData[0:4], base=2)
        dtime = int(recordData[4:16], base=2)
        nsync = int(recordData[16:32], base=2)
                
        if channel == 0xF: # Special record
        
            if dtime == 0: # Not a marker, so overflow
                # print("%u OFL * %2x\n" % (recNum, 1))
                oflcorrection += T3WRAPAROUND
                
            else: # got marker
                truensync = oflcorrection + nsync
                print("%u MAR %2x %u\n" % (recNum, truensync, dtime))
                
        else: # standard record, photon count
            
            if channel == 0 or channel > 4: # Should not occur
                print("Illegal Channel: #%1d %1u" % (dlen, channel))
                
            truensync = oflcorrection + nsync
            
            dtime_array[recNum] = dtime
            truensync_array[recNum] = truensync
            
            dlen += 1
            
        # if recNum % 100000 == 0:
        #     sys.stdout.write("\rProgress: %.1f%%" % (float(recNum)*100/float(numRecords)))
        #     sys.stdout.flush()
    tf = time.time()
    print(tf-t0, channel)

    return dtime_array, truensync_array

# import numba
# @numba.njit
def readPT3bis(inputfile, numRecords):

    bdata = memoryview(inputfile.read()).cast("I")

    t0 = time.time()
    rv = inner(bdata)
    tf = time.time()
    print(tf-t0)
    return rv


@numba.njit(numba.types.UniTuple(numba.int64[:], 2)
            (numba.typeof(memoryview(bytes([0]*4)).cast("I"))),
            parallel=True)
def inner(data):
    oflcorrection = 0
    T3WRAPAROUND = 65536

    numRecords = len(data)
    dtime_array = np.zeros(numRecords, dtype=np.int64)
    truensync_array = np.zeros(numRecords, dtype=np.int64)
    recnum = 0
    for d in data:
        nsync = d & 0b1111111111111111
        d = d >> 16
        dtime = d & 0b111111111111
        d = d >> 12
        channel = d  # & 0b1111 No necesario
        if channel == 0xF:  # Special record
            if dtime == 0:  # Not a marker, so overflow
                oflcorrection += T3WRAPAROUND
            else:  # got marker
                truensync = oflcorrection + nsync
        else:  # standard record, photon count
            truensync = oflcorrection + nsync
            dtime_array[recnum] = dtime
            truensync_array[recnum] = truensync
            recnum += 1
    return dtime_array[:recnum], truensync_array[:recnum]
# def inner(bdata):
#     oflcorrection = 0
#     dlen = 0
#     T3WRAPAROUND = 65536

#     numRecords = len(bdata)
#     dtime_array = np.zeros(numRecords)
#     truensync_array = np.zeros(numRecords)

#     for recNum, data in enumerate(bdata):
#         nsync = data & 0b1111111111111111
#         data = data >> 16
#         dtime = data & 0b111111111111
#         data = data >> 12
#         channel = data  # & 0b1111 No necesario
#         if channel == 0xF:  # Special record
#             if dtime == 0:  # Not a marker, so overflow
#                 oflcorrection += T3WRAPAROUND
#             else:  # got marker
#                 truensync = oflcorrection + nsync
#         else:  # standard record, photon count
#             truensync = oflcorrection + nsync
#             dtime_array[recNum] = dtime
#             truensync_array[recNum] = truensync
#             dlen += 1
#     # return dtime_array, truensync_array
#     return dtime_array[:dlen], truensync_array[:dlen]

@numba.njit
def bintest(timerel: np.ndarray):
    """binnea por índice, no por tiempo."""
    binwidth = 400  # a sacar de la configuración
    nbins = 4
    max_v = binwidth * nbins
    # esperamos valores desde 0 .. max_v-1
    result = np.zeros((nbins,), dtype=np.uint64)
    # print(result)
    for x in timerel:
        result[x // binwidth] += 1
    return result

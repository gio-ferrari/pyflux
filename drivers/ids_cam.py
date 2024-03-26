# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:04:06 2023

@author: Florencia D. Choque 
Script for taking pictures using IDS Industrial Camera U3-3060CP-M-GL Rev.2.2
Win 10 - 64 bit
Full cheap 1920x1200 
It's better to revoke buffers in a separate function as it is done here.
The sensor gives us a bayered image that can be debayered using ids_peak_ipl
The converted image can then be processed to convert raw data to image
The faster process: taking just 1 channel
User Manual: https://de.ids-imaging.com/manuals/ids-peak/ids-peak-user-manual/1.3.0/en/basics-bayer-pattern.html
I created this class to introduce camera as an IDS object in a backend class (another program, bigger), otherwise I had problems with threading, on_acquisition_timer can be modified to return 1 channel image
When running in stand alone mode device.work() gives a camera photo
"""

import sys
from ids_peak import ids_peak
from ids_peak_ipl import ids_peak_ipl 
import matplotlib.pyplot as plt
from ids_peak import ids_peak_ipl_extension
import numpy as np
import time

FPS_LIMIT = 30

class IDS_U3:
    def __init__(self):
        
        self.__device = None
        self.__nodemap_remote_device = None
        self.__datastream = None
        
        #Variables de instancia relacionadas con laadquisición de imágenes
        self.__display = None
        #self.__acquisition_timer = QTimer() #temporizador para controlar la frecuencia de adquisición,
        self.counter = 0 #Contador del numero de cuadros
        self.__error_counter = 0 #Contador del numero de errores
        self.__acquisition_running = False #bandera para indicar si la adquisición está en curso.
        
        ids_peak.Library.Initialize() 
 
    def destroy_all(self):
        # Stop acquisition
        self.stop_acquisition()

        # Close device and peak library
        self.close_device()
        ids_peak.Library.Close()
     
    def open_device(self):
        try:
            # Create instance of the device manager
            device_manager = ids_peak.DeviceManager.Instance()

            # Update the device manager
            device_manager.Update()

            # Return if no device was found
            if device_manager.Devices().empty():
                print( "Error", "No device found!")
                return False
            else:
                print("Device found")

            # Open the first openable device in the managers device list
            for device in device_manager.Devices():
                if device.IsOpenable():
                    self.__device = device.OpenDevice(ids_peak.DeviceAccessType_Control)
                    break

            # Return if no device could be opened
            if self.__device is None:
                print( "Error", "Device could not be opened!")
                return False

            # Open standard data stream
            datastreams = self.__device.DataStreams()
            if datastreams.empty():
                print("Error", "Device has no DataStream!")
                self.__device = None
                return False

            self.__datastream = datastreams[0].OpenDataStream()

            # Get nodemap of the remote device for all accesses to the genicam nodemap tree
            self.__nodemap_remote_device = self.__device.RemoteDevice().NodeMaps()[0]

            # To prepare for untriggered continuous image acquisition, load the default user set if available and
            # wait until execution is finished
            try:
                self.__nodemap_remote_device.FindNode("UserSetSelector").SetCurrentEntry("Default")
                self.__nodemap_remote_device.FindNode("UserSetLoad").Execute()
                self.__nodemap_remote_device.FindNode("UserSetLoad").WaitUntilDone()
            except ids_peak.Exception:
                # Userset is not available
                pass
            #Setting exposure time
            
            min_exposure_time = 0
            max_exposure_time = 0
            exposure_time_value = 50000.0 #us
 
            # Get exposure range. All values in microseconds
            min_exposure_time = self.__nodemap_remote_device.FindNode("ExposureTime").Minimum()
            max_exposure_time = self.__nodemap_remote_device.FindNode("ExposureTime").Maximum()
            #print("Min_exposure_time: ", min_exposure_time, "us.") #Min_exposure_time:  28.527027027027028 us
            #print("Max exposure time: ", max_exposure_time, "us.") # Max exposure time:  2000001.6351351351 us = 2000 ms = 2 s
            # if self.__nodemap_remote_device.FindNode("ExposureTime").HasConstantIncrement():
            #      inc_exposure_time = self.__nodemap_remote_device.FindNode("ExposureTime").Increment()
            # else:
            #     # If there is no increment, it might be useful to choose a suitable increment for a GUI control element (e.g. a slider)
            #      inc_exposure_time = 1000
 
            # Get the current exposure time
            exposure_time = self.__nodemap_remote_device.FindNode("ExposureTime").Value() #This is a default value
            #print("Current exposure time: ", exposure_time, "us.", type(exposure_time)) #Current exposure time:  14998.45945945946 us = 15ms
            # Set exposure time to exposure_time_value
            self.__nodemap_remote_device.FindNode("ExposureTime").SetValue(exposure_time_value)
            exposure_time = self.__nodemap_remote_device.FindNode("ExposureTime").Value() #This is a default value
            print("NEW Current exposure time: ", exposure_time/1000.0, "ms.")
            return True
        
        except ids_peak.Exception as e:
            print( "Exception", str(e))

            return False
        
    def set_roi(self,x, y, width, height):
        try:
            # Get the minimum ROI and set it. After that there are no size restrictions anymore
            x_min = self.__nodemap_remote_device.FindNode("OffsetX").Minimum()
            #print("x_min:", x_min)
            y_min = self.__nodemap_remote_device.FindNode("OffsetY").Minimum()
            #print("y_min:", y_min)
            w_min = self.__nodemap_remote_device.FindNode("Width").Minimum()
            #print("w_min:", w_min)
            h_min = self.__nodemap_remote_device.FindNode("Height").Minimum()
            #print("h_min:", h_min)
     
            self.__nodemap_remote_device.FindNode("OffsetX").SetValue(x_min)
            self.__nodemap_remote_device.FindNode("OffsetY").SetValue(y_min)
            self.__nodemap_remote_device.FindNode("Width").SetValue(w_min)
            self.__nodemap_remote_device.FindNode("Height").SetValue(h_min)
     
            # Get the maximum ROI values
            x_max = self.__nodemap_remote_device.FindNode("OffsetX").Maximum()
            #print("x_max:", x_max)
            y_max = self.__nodemap_remote_device.FindNode("OffsetY").Maximum()
            #print("y_max:", y_max)
            w_max = self.__nodemap_remote_device.FindNode("Width").Maximum()
            #print("w_max:", w_max)
            h_max = self.__nodemap_remote_device.FindNode("Height").Maximum()
            #print("h_max:", h_max)
     
            if (x < x_min) or (y < y_min) or (x > x_max) or (y > y_max):
                return False
            elif (width < w_min) or (height < h_min) or ((x + width) > w_max) or ((y + height) > h_max):
                return False
            else:
                # Now, set final AOI
                self.__nodemap_remote_device.FindNode("OffsetX").SetValue(x)
                self.__nodemap_remote_device.FindNode("OffsetY").SetValue(y)
                self.__nodemap_remote_device.FindNode("Width").SetValue(width)
                self.__nodemap_remote_device.FindNode("Height").SetValue(height)
                return True
        except Exception as e:
            str_error = str(e)
            print("Error setting roi: ", str_error)
            return False
         
    def alloc_and_announce_buffers(self):
        try:
            if self.__datastream:
                self.revoke_buffers()
                payload_size = self.__nodemap_remote_device.FindNode("PayloadSize").Value()
                #print(payload_size, "-> Payload_size") # Payload_size in this case is: 65536 bytes, for width = 256 and height = 256 # 2304000 -> Payload_size for full chip (width:1920, height: 1200)
     
                # Get number of minimum required buffers
                num_buffers_min_required = self.__datastream.NumBuffersAnnouncedMinRequired() #num_buffers_min_required:  3
     
                # Alloc buffers
                for i in range(num_buffers_min_required):
                    try:
                        buffer = self.__datastream.AllocAndAnnounceBuffer(payload_size)
                        self.__datastream.QueueBuffer(buffer)
                    except Exception as e:
                        print("Error en parte nueva de buffers: ", str(e))
                return True
        except Exception as e:
            str_error = str(e)
            print("Error in buffers: ", str_error)
            return False
                
    def revoke_buffers(self):
        try:
            if self.__datastream:
                # Flush queue and prepare all buffers for revoking
                self.__datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
     
                # Clear all old buffers
                for buffer in self.__datastream.AnnouncedBuffers():
                    self.__datastream.RevokeBuffer(buffer)
                return True
        except Exception as e:
            print("Error revoking buffers: ", str(e))
            return False     
    
    def close_device(self):
        """
        Stop acquisition if still running and close datastream and nodemap of the device
        """
        # Stop Acquisition in case it is still running
        self.stop_acquisition()

        # If a datastream has been opened, try to revoke its image buffers
        if self.__datastream is not None:
            try:
                for buffer in self.__datastream.AnnouncedBuffers():
                    self.__datastream.RevokeBuffer(buffer)
            except Exception as e:
                print("Exception", str(e))
    
    def start_acquisition(self):
        """
        Start Acquisition on camera and start the acquisition timer to receive and display images

        :return: True/False if acquisition start was successful
        """
        # Check that a device is opened and that the acquisition is NOT running. If not, return.
        if self.__device is None:
            return False
        if self.__acquisition_running is True:
            return True

        # Get the maximum framerate possible, limit it to the configured FPS_LIMIT. If the limit can't be reached, set
        # acquisition interval to the maximum possible framerate
        try:
            max_fps = self.__nodemap_remote_device.FindNode("AcquisitionFrameRate").Maximum() #Max Frame Rate:  66.49384258032052 FPS_LIMIT:  30
            #print("Max Frame Rate: ", max_fps, "FPS_LIMIT: ", FPS_LIMIT)
            target_fps = min(max_fps, FPS_LIMIT)
            self.__nodemap_remote_device.FindNode("AcquisitionFrameRate").SetValue(target_fps)
        except ids_peak.Exception:
            # AcquisitionFrameRate is not available. Unable to limit fps. Print warning and continue on.
            print( "Warning: Unable to limit fps, since the AcquisitionFrameRate Node is not supported by the connected camera. Program will continue without limit.")

        # Setup acquisition timer accordingly
        #self.cameraTimer.setInterval((1 / target_fps) * 1000) #Same timer than in uc480
        #self.cameraTimer.setSingleShot(False)
        #self.cameraTimer.timeout.connect(self.on_acquisition_timer)# Important line when working with no Thread
        try:
            # Lock critical features to prevent them from changing during acquisition
            self.__nodemap_remote_device.FindNode("TLParamsLocked").SetValue(1)

            # Start acquisition on camera
            self.__datastream.StartAcquisition()
            self.__nodemap_remote_device.FindNode("AcquisitionStart").Execute()
            self.__nodemap_remote_device.FindNode("AcquisitionStart").WaitUntilDone()

        except Exception as e:
            print("Exception: " + str(e))
            return False

        # Start acquisition timer
        #self.cameraTimer.start()
        self.__acquisition_running = True

        return True
    
    def stop_acquisition(self):
        """
        Stop acquisition timer and stop acquisition on camera
        :return:
        """
        # Check that a device is opened and that the acquisition is running. If not, return.
        if self.__device is None or self.__acquisition_running is False:
            return

        # Otherwise try to stop acquisition
        try:
            remote_nodemap = self.__device.RemoteDevice().NodeMaps()[0]
            remote_nodemap.FindNode("AcquisitionStop").Execute()

            # Stop and flush datastream
            self.__datastream.KillWait()
            self.__datastream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
            self.__datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)

            self.__acquisition_running = False

            # Unlock parameters after acquisition stop
            if self.__nodemap_remote_device is not None:
                try:
                    self.__nodemap_remote_device.FindNode("TLParamsLocked").SetValue(0)
                except Exception as e:
                    print("Exception", str(e))

        except Exception as e:
            print("Exception", str(e))

    def on_acquisition_timer(self): #this is show_image function
        try:
            # Get buffer from device's DataStream. Wait 5000 ms. The buffer is automatically locked until it is queued again.
            if self.__datastream:
                # Get buffer from device's datastream
                buffer = self.__datastream.WaitForFinishedBuffer(5000)
                ipl_image = ids_peak_ipl_extension.BufferToImage(buffer)
                converted_ipl_image = ipl_image.ConvertTo(ids_peak_ipl.PixelFormatName_BGRa8, ids_peak_ipl.ConversionMode_Fast)

                # Queue buffer again
                self.__datastream.QueueBuffer(buffer)
                
                # Get raw image data from converted image and construct a 3D array
                self.image_np_array = converted_ipl_image.get_numpy_3D()
                image_sum = self.image_np_array[:, :, 0]#R channel
                
                # 2D array, each element is the sum of the R,G,B,A channels
                #image_sum = np.sum(self.image_np_array, axis=2)


                #plt.imshow(self.image_red)
                return image_sum.copy() ## Make an extra copy of the to make sure that memory is copied and can't get overwritten later on, this is also done in uc480
        
        except Exception as e:
            print("Error showing image: ", str(e))

     
    def work(self):     
        if self.open_device():
            # error
            print("Success opening device")
        if not self.set_roi(16, 16, 1920, 1200): # full chip (width:1920, height: 1200)
            # error
            sys.exit(-2)
        if not self.alloc_and_announce_buffers():
            # error
            sys.exit(-3)
        if not self.start_acquisition():
            # error
            sys.exit(-4)
        #sys.exit(0)

     
if __name__ == '__main__':
    
    device = IDS_U3()
    
    value = device.open_device()
    if value == True:
        print("Big success")
    device.work()
    start=time.perf_counter()
    image= device.on_acquisition_timer()
    end=time.perf_counter()
    print("Time on_acquisition_timer execution: ", end-start) #time show_image() execution:  0.05100199999999866
    #print(type(image)) #<class 'numpy.ndarray'>
    plt.imshow(image)
    sys.exit(0)



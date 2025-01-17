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
https://de.ids-imaging.com/manuals/ids-peak/ids-peak-user-manual/1.3.0/en/program-intro.html
I created this class to introduce camera as an IDS object in a backend class (another program, bigger), otherwise I had problems with threading,
on_acquisition_timer function can be modified to return 1 channel image
When running in stand alone mode device.take_image() displays a camera photo.
TriggerSoftware:To trigger the capture of an image.
TODO: Add gain control

"""

import sys
from ids_peak import ids_peak
from ids_peak_ipl import ids_peak_ipl
import matplotlib.pyplot as plt
from ids_peak import ids_peak_ipl_extension
import time
import logging as _lgn

_lgn.basicConfig(level=_lgn.INFO,
                 format='%(asctime)s:%(name)s:%(levelname)s:%(message)s',
                 )
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)

FPS_LIMIT = 30.0  # Hz, FPS: Acquisition Frame Rate to meet the desirable application's speed.
EXPOSURE_TIME_VALUE = 50000.0  # µs # Default exposure time


class IDS_U3:
    """IDS Camera."""

    def __init__(self, exposure_time = EXPOSURE_TIME_VALUE):
        """Init Camera."""
        self.exposure_time = exposure_time
        self.__device = None
        self.__nodemap_remote_device = None
        self.__datastream = None
        # Variables de instancia relacionadas con la adquisición de imágenes
        self.__display = None
        self.__error_counter = 0  # Contador del numero de errores
        self.__acquisition_running = False  # bandera para indicar si la adquisición está en curso.
        ids_peak.Library.Initialize()

    def destroy_all(self):
        # Stop acquisition
        self.stop_acquisition()
        # Close device and peak library
        self.close_device()
        ids_peak.Library.Close()

    def open_device(self):
        # https://www.1stvision.com/cameras/IDS/IDS-manuals/en/pixel-format.html
        try:
            # Create instance of the device manager
            device_manager = ids_peak.DeviceManager.Instance()

            # Update the device manager
            device_manager.Update()

            # Return if no device was found
            if device_manager.Devices().empty():
                _lgr.error("No device found!")
                return False
            _lgr.debug("Device found")

            # Open the first openable device in the managers device list
            for device in device_manager.Devices():
                if device.IsOpenable():
                    self.__device = device.OpenDevice(ids_peak.DeviceAccessType_Control)
                    break

            # Return if no device could be opened
            if self.__device is None:
                _lgr.error("Device could not be opened!")
                return False
            _lgr.debug("Device opened: %s (S/N %s)", self.__device.DisplayName(),
                       self.__device.SerialNumber())

            # Open standard data stream
            datastreams = self.__device.DataStreams()
            if datastreams.empty():
                _lgr.error("Device has no DataStream!")
                self.__device = None
                return False

            self.__datastream = datastreams[0].OpenDataStream()

            # Get nodemap of the remote device for all accesses to the genicam nodemap tree
            self.__nodemap_remote_device = self.__device.RemoteDevice().NodeMaps()[0]

            # To prepare for untriggered continuous image acquisition, load the
            # default user set if available and wait until execution is finished.
            #Default (freerun mode): The camera captures image frames continuously with the given frame rate
            #When you load UserSet "Default", all other parameters are set to their default value, too. Therefore use this method before you change any other parameters.
            # https://www.1stvision.com/cameras/IDS/IDS-manuals/en/user-set-selector.html
            try:
                self.__nodemap_remote_device.FindNode("UserSetSelector").SetCurrentEntry("Default")
                acq_mode = self.__nodemap_remote_device.FindNode("AcquisitionMode").CurrentEntry().SymbolicValue()
                _lgr.info("Acquisition Mode in Default: %s", acq_mode)
                self.__nodemap_remote_device.FindNode("UserSetLoad").Execute()
                self.__nodemap_remote_device.FindNode("UserSetLoad").WaitUntilDone()
            except ids_peak.Exception:
                # Userset is not available
                _lgr.info("UserSet not available")
                pass
            # FIXME: Test Triggered Mode
            self.__nodemap_remote_device.FindNode("TriggerSelector").SetCurrentEntry("ExposureStart")
            self.__nodemap_remote_device.FindNode("TriggerSource").SetCurrentEntry("Software")
            self.__nodemap_remote_device.FindNode("TriggerMode").SetCurrentEntry("On")
            
            self.set_exposure_time(self.exposure_time)
            try:
                ds = self.__datastream.NodeMaps()
                ds = ds[0]
                _lgr.info("Modo buffering camara: %s",
                            ds.FindNode("StreamBufferHandlingMode").CurrentEntry().SymbolicValue())
                # # Sólo para debug inicial, borrar luego
                # allEntries = ds.FindNode("StreamBufferHandlingMode").Entries()
                # availableEntries = []
                # for entry in allEntries:
                #     if (entry.AccessStatus() != ids_peak.NodeAccessStatus_NotAvailable
                #             and entry.AccessStatus() != ids_peak.NodeAccessStatus_NotImplemented):
                #         availableEntries.append(entry.SymbolicValue())
                # _lgr.info("Modos disponibles: %s", availableEntries)
                ds.FindNode("StreamBufferHandlingMode").SetCurrentEntry("NewestOnly")
                _lgr.info("Modo buffering camara actual: %s",
                            ds.FindNode("StreamBufferHandlingMode").CurrentEntry().SymbolicValue())
            except Exception as e:
                _lgr.error("Error seteando modo de buffering de la cámara: %s", e)
            # FIXME: Hardcoded values to set ROI, 16 seems to be the number that work
            if not self.set_roi(16, 16, 1920, 1200):
                _lgr.error("Error setting ROI")
                return
            if not self.alloc_and_announce_buffers():
                _lgr.error("Error with buffers")
                return
            return True
        except ids_peak.Exception as e:
            _lgr.error("Exception %s: %s", type(e), str(e))
            return False
        
    def set_exposure_time(self, exposure_time):
                    # Setting exposure time
            min_exposure_time = 0
            max_exposure_time = 0
            self.exposure_time = exposure_time
            # Get exposure range. All values in microseconds
            min_exposure_time = self.__nodemap_remote_device.FindNode("ExposureTime").Minimum() # Min exposure_time:  28.527027027027028 us
            max_exposure_time = self.__nodemap_remote_device.FindNode("ExposureTime").Maximum() # Max exposure time:  2000001.6351351351 us = 2000 ms = 2 s
            #_lgr.info("Min / Max exposure time: %s µs / %s µs", min_exposure_time, max_exposure_time)
            # if self.__nodemap_remote_device.FindNode("ExposureTime").HasConstantIncrement():
            #      inc_exposure_time = self.__nodemap_remote_device.FindNode("ExposureTime").Increment()
            # else:
            #     # If there is no increment, it might be useful to choose a suitable increment for a GUI control element (e.g. a slider)
            #      inc_exposure_time = 1000

            # Set exposure time to exposure_time
            self.__nodemap_remote_device.FindNode("ExposureTime").SetValue(self.exposure_time)
            exp_time = self.__nodemap_remote_device.FindNode("ExposureTime").Value() #This is a default value
            _lgr.info("New Current exposure time: %s ms.", exp_time/1E3)

    
    def set_roi(self, x, y, width, height):
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
                _lgr.warning("selected ROI origin outside bounds")
                return False
            elif ((width < w_min) or (height < h_min) or ((x + width) > w_max)
                  or ((y + height) > h_max)):
                _lgr.warning("selected ROI dimensions outside bounds")
                return False
            else:
                # Now, set final ROI
                self.__nodemap_remote_device.FindNode("OffsetX").SetValue(x)
                self.__nodemap_remote_device.FindNode("OffsetY").SetValue(y)
                self.__nodemap_remote_device.FindNode("Width").SetValue(width)
                self.__nodemap_remote_device.FindNode("Height").SetValue(height)
                return True
        except Exception as e:
            _lgr.error("Exception setting roi %s: %s", type(e), e)
            return False

    def alloc_and_announce_buffers(self):
        try:
            if self.__datastream:
                self.revoke_buffers()
                payload_size = self.__nodemap_remote_device.FindNode("PayloadSize").Value()
                num_buffers_min_required = self.__datastream.NumBuffersAnnouncedMinRequired()
                _lgr.debug("Camera payload size is %s, and a minimum of %s buffers are required", payload_size, num_buffers_min_required)
                # Payload_size in this case is: 65536 bytes, for width = 256 and height = 256 # 2304000 -> Payload_size for full chip (width:1920, height: 1200)
                #num_buffers_min_required:  3

                # Alloc buffers
                for i in range(num_buffers_min_required):
                    try:
                        buffer = self.__datastream.AllocAndAnnounceBuffer(payload_size)
                        self.__datastream.QueueBuffer(buffer)
                    except Exception as e:
                        _lgr.error("Error allocating and setting buffers: %s", str(e))
                return True
        except Exception as e:
            _lgr.error("Exception allocating buffers: %s (%s)", type(e), e)
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
            _lgr.error("Exception revoking buffers: %s (%s)", type(e), e)
            return False

    def close_device(self):
        """
        Stop acquisition if still running and close datastream and nodemap of the device
        """
        # Stop Acquisition in case it is still running
        self.stop_acquisition()
        # If a datastream has been opened, try to revoke its image buffers
        self.revoke_buffers()

    def start_acquisition(self):
        """
        Start Acquisition on camera and start the acquisition timer to receive and display images

        return: True/False if acquisition start was successful
        """
        # Check that a device is opened and that the acquisition is NOT running. If not, return.
        # https://www.1stvision.com/cameras/IDS/IDS-manuals/en/acquisition-mode.html
        if self.__device is None:
            return False
        if self.__acquisition_running is True:
            return True

        # Get the maximum framerate possible, limit it to the configured
        # FPS_LIMIT. If the limit can't be reached, set acquisition interval
        # to the maximum possible framerate
        try:
            max_fps = self.__nodemap_remote_device.FindNode("AcquisitionFrameRate").Maximum() #Max Frame Rate:  66.49384258032052 FPS_LIMIT:  30
            old_fps_value = self.__nodemap_remote_device.FindNode("AcquisitionFrameRate").Value()
            target_fps = min(max_fps, FPS_LIMIT)
            self.__nodemap_remote_device.FindNode("AcquisitionFrameRate").SetValue(target_fps)
            new_fps_value = self.__nodemap_remote_device.FindNode("AcquisitionFrameRate").Value()
            _lgr.info("Old frame rate: %s fps. New frame rate: %s fps.", old_fps_value, new_fps_value)
        except ids_peak.Exception:
            # AcquisitionFrameRate is not available. Unable to limit fps. Print warning and continue on.
            _lgr.warning("Unable to limit fps: AcquisitionFrameRate Node is not supported by the connected camera. Program will continue without limit.")

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
            _lgr.error("Error starting acquisition: %s", e, str(e))
            return False

        # Start acquisition timer
        #self.cameraTimer.start()
        self.__acquisition_running = True

        return True

    def stop_acquisition(self):
        """Stop acquisition."""
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
                    _lgr.error("Exception: %s", e, str(e))

        except Exception as e:
            _lgr.error("Exception: %s", e, str(e))

    def on_acquisition_timer(self): #this is show_image function for uc480
        try:
	# FIXME: test para triggered
            self.__nodemap_remote_device.FindNode("TriggerSoftware").Execute()
            # Get buffer from device's DataStream. Wait 5000 ms. The buffer is automatically locked until it is queued again.
            if self.__datastream:
                # Get buffer from device's datastream
                buffer = self.__datastream.WaitForFinishedBuffer(500) #IDS Manual: 5000, Try 500
                ipl_image = ids_peak_ipl_extension.BufferToImage(buffer)
                # PixelFormatName_Mono8???  PixelFormatName_BGRa8
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
            _lgr.error("Exception acquiring image: %s (%s)", e, str(e))
            raise

    def take_image(self):
        """Test Function."""
        self.start_acquisition()
        start=time.perf_counter()
        image= self.on_acquisition_timer() # Returns an image
        #print(type(image)) #<class 'numpy.ndarray'>
        end=time.perf_counter()
        _lgr.debug("Time on_acquisition_timer execution: %s s.", end-start)
        plt.imshow(image)
        _lgr.info("Success taking image")
        return


if __name__ == '__main__':
    device = IDS_U3()
    if device.open_device():
        print("Big success, device configured")
    else:
        print("Error opening device")
    try:
        device.take_image()
    finally:
        device.destroy_all()



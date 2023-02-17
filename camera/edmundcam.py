from pyueye import ueye  # for using iDS uEye camera
import numpy as np

class EdmundCamera:
    def __init__(self, make_square: bool = False, gain: int = 100, gamma: int = 100, fps: float = 60.0, exposure_time: float = 0.001):
        pnNumCams = ueye.int()  # Container for number of camera(s)
        self.hCam = ueye.HIDS(0)  # 0:first available camera; 1-254:The camera with the specified camera ID; The factory default camera ID is 1. The camera ID can also be changed in the IDS Camera Manager.
        cInfo = ueye.CAMINFO()  # Container for camera info
        sInfo = ueye.SENSORINFO()  # Container for sensor info
        m_nColorMode = ueye.INT()  # Container for camera color mode Y8/RGB16/RGB24/REG32
        self.image_memory = ueye.c_mem_p()  # Pointer to the memory starting address
        MemID = ueye.int()
        self.pitch = ueye.INT()

        ## Initialize the camera
        if ueye.is_GetNumberOfCameras(pnNumCams) != ueye.IS_SUCCESS:
            print("iDS uEye camera 'is_GetNumberOfCameras' ERROR")
            self.is_connected = False
        print("Found %s iDS uEye Camera(s)" % pnNumCams)

        ## Starts the driver and establishes the connection to the camera; "None" for Bitmap (DIB) mode
        if ueye.is_InitCamera(self.hCam, None) != ueye.IS_SUCCESS:
            print("iDS uEye camera 'is_InitCamera' ERROR")
            self.is_connected = False

        ## Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
        if ueye.is_GetCameraInfo(self.hCam, cInfo) != ueye.IS_SUCCESS:
            print("iDS uEye camera 'is_GetCameraInfo' ERROR")
            self.is_connected = False
        print(f"{cInfo.ID.decode('utf-8')}  SerNo:{cInfo.SerNo.decode('utf-8')}")


        ## You can query additional information about the sensor type used in the camera
        if ueye.is_GetSensorInfo(self.hCam, sInfo) != ueye.IS_SUCCESS:
            print("iDS uEye camera 'is_GetSensorInfo' ERROR")
        print(f"SensorName: {sInfo.strSensorName.decode('utf-8')}  SensorID: {sInfo.SensorID}")
        self._N = int(sInfo.nMaxHeight)  # It is uint. Change it to int for setting AOI
        self._M = int(sInfo.nMaxWidth)
        self._lp = sInfo.wPixelSize * 1e-8  # in [m]
        self._height = self._N * self._lp  # in [m]
        self._width = self._M * self._lp  # in [m]
        nColorMode = int.from_bytes(sInfo.nColorMode.value, byteorder='big')  # only one byte, big or little doesn't matter
        shutter_dict = {0: "Shutter: rolling", 1: "Shutter: global"}
        color_mode_dict = {1: "IS_COLORMODE_MONOCHROME", 2: "IS_COLORMODE_BAYER", 4: "IS_COLORMODE_CBYCRY (USB uEye XS only)", 8: "IS_COLORMODE_JPEG (USB uEye XS only)", 0: "IS_COLORMODE_INVALID"}
        print(shutter_dict[int(sInfo.bGlobShutter)])
        print("Color mode: ", color_mode_dict[int(nColorMode)])

        ## Resets all parameters to the camera-specific defaults as specified by the driver. By default, the camera uses full resolution, a medium speed and color level gain values adapted to daylight exposure.
        if ueye.is_ResetToDefault(self.hCam) != ueye.IS_SUCCESS:
            print("iDS uEye camera 'is_ResetToDefault' ERROR")

        ## Set display mode to Bitmap (DIB) mode
        if ueye.is_SetDisplayMode(self.hCam, ueye.IS_SET_DM_DIB) != ueye.IS_SUCCESS:
            print("iDS uEye camera 'is_SetDisplayMode' ERROR")

        ## Set the right color mode and pixel bit depth (Be careful, even if you set the wrong one, no error occur)
        self.n_bits_per_pixel = ueye.INT(10)  ##### Edit it if needed #####
        m_nColorMode = ueye.IS_CM_SENSOR_RAW10   ##### Edit it if needed #####
        self.n_channels = ueye.INT(1)
        if ueye.is_SetColorMode(self.hCam, m_nColorMode) != ueye.IS_SUCCESS:
            print("iDS uEye camera 'is_SetColorMode' ERROR")
        print(f"Pixel bit depth: {self.n_bits_per_pixel} bits ")

        # ## Set image capture parameters

        ## Make the image to be square. This will lose a little information, but can make frequency domain (kx & ky) spacing equally
        if make_square:
            self._N, self._M = 2048, 2048 # UI388xCP-M is 1308x2076. But it doesn't support 2076x2076, only support 2072x2072. I choose 2048 for faster fft.
            set_rectAOI = ueye.IS_RECT(520, 14, self._M, self._N)
            nRet = ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_SET_AOI, set_rectAOI, ueye.sizeof(set_rectAOI))
            if nRet != ueye.IS_SUCCESS:
                print("is_AOI ERROR")
            print(f"set AOI X position: {set_rectAOI.s32X}")
            print(f"set AOI y position: {set_rectAOI.s32Y}")
            print(f"set AOI width:      {set_rectAOI.s32Width}")
            print(f"set AOI height:     {set_rectAOI.s32Height}")
        rectAOI = ueye.IS_RECT()
        nRet = ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_GET_AOI, rectAOI, ueye.sizeof(rectAOI))
        if nRet != ueye.IS_SUCCESS:
            print("is_AOI ERROR")
        print(f"AOI X position: {rectAOI.s32X}")
        print(f"AOI y position: {rectAOI.s32Y}")
        print(f"AOI width:      {rectAOI.s32Width}")
        print(f"AOI height:     {rectAOI.s32Height}")
        self.aoi_width = rectAOI.s32Width
        self.aoi_height = rectAOI.s32Height

        self.gain = gain
        self.gamma = gamma

        # Set camera pixel clock [MHz]
        # (UI388xCP-M, MONO-8bit mode support pixel clock support 118 or 237 or 474 MHz, MONO-12bit mode support 99 or 197 MHz)
        # (UI324xCP-M, MONO-10bit mode support pixel clock max 86 MHz)
        set_cam_pixel_clock = ueye.UINT(19)
        cam_pixel_clock = ueye.UINT()
        nRet = ueye.is_PixelClock(self.hCam, ueye.IS_PIXELCLOCK_CMD_SET, set_cam_pixel_clock, ueye.sizeof(set_cam_pixel_clock))
        if nRet != ueye.IS_SUCCESS:
            print("is_PixelClock ERROR")
        print(f'Set Pixel Clock: \t{set_cam_pixel_clock} MHz')
        nRet = ueye.is_PixelClock(self.hCam, ueye.IS_PIXELCLOCK_CMD_GET, cam_pixel_clock, ueye.sizeof(cam_pixel_clock))
        if nRet != ueye.IS_SUCCESS:
            print("is_PixelClock ERROR")
        print(f'Current Pixel Clock: \t{cam_pixel_clock} MHz')

        self.fps = fps
        self.exposure_time = exposure_time


        # ## Allocates image memory

        # Allocates an image memory for an image having its dimensions defined by width and height and its color depth defined by nBitsPerPixel
        nRet = ueye.is_AllocImageMem(self.hCam, self.aoi_width, self.aoi_height, self.n_bits_per_pixel, self.image_memory, MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_AllocImageMem ERROR")
        else:
            # Makes the specified image memory the active memory
            nRet = ueye.is_SetImageMem(self.hCam, self.image_memory, MemID)
            if nRet != ueye.IS_SUCCESS:
                print("is_SetImageMem ERROR")
            else:
                # Set the desired color mode
                nRet = ueye.is_SetColorMode(self.hCam, m_nColorMode)

        # Activates the camera's software trigger mode (Under this mode, there are is_CaptureVideo() live function
        # and is_FreezeVideo() snap function)
        nRet = ueye.is_SetExternalTrigger(self.hCam, ueye.IS_SET_TRIGGER_SOFTWARE)  # IS_SET_TRIGGER_SOFTWARE or IS_SET_TRIGGER_OFF
        if nRet != ueye.IS_SUCCESS:
            print("is_SetExternalTrigger ERROR")
        nRet = ueye.is_SetExternalTrigger(self.hCam, ueye.IS_GET_EXTERNALTRIGGER)
        is_SetExternalTrigger_return = {ueye.IS_SET_TRIGGER_OFF: 'IS_SET_TRIGGER_OFF',
                                        ueye.IS_SET_TRIGGER_SOFTWARE: 'IS_SET_TRIGGER_SOFTWARE'}
        print(is_SetExternalTrigger_return[nRet])

        # Reads out the properties of an allocated image memory
        nRet = ueye.is_InquireImageMem(self.hCam, self.image_memory, MemID, self.aoi_width, self.aoi_height, self.n_bits_per_pixel, self.pitch)
        if nRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")

        # Capture one image in to the memory
        nRet = ueye.is_FreezeVideo(self.hCam, ueye.IS_WAIT)
        if nRet != ueye.IS_SUCCESS:
            print("is_FreezeVideo ERROR", nRet)
        print("Camera setup")

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, value: float):
        # Set camera frame rate [FPS]
        self._fps = ueye.DOUBLE(value)
        cam_frame_rate = ueye.DOUBLE()
        nRet = ueye.is_SetFrameRate(self.hCam, self._fps, cam_frame_rate)
        if nRet != ueye.IS_SUCCESS:
            print("is_SetFrameRate ERROR")
        print('Set FrameRate (FPS): \t', self._fps)
        print('Current FrameRate (FPS):', cam_frame_rate)

    @property
    def exposure_time(self):
        return self._exposure_time

    @exposure_time.setter
    def exposure_time(self, value: float):
        # Set camera exposure time [ms]
        # w/o ND 3.573 # UI388xCP-M 12bit min is 0.021 ms. # After setting the exposure time, this value contains the actually set exposure time
        # UI324xCP-M, MONO-12bit min is 0.009 ms
        self._exposure_time = ueye.DOUBLE(value)  #
        print('Set Exposure time [ms]: \t', self._exposure_time)
        nRet = ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, self._exposure_time,
                                ueye.sizeof(self._exposure_time))
        if nRet != ueye.IS_SUCCESS:
            print("is_Exposure ERROR")
        print('Current Exposure time [ms]: \t', self._exposure_time)

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value: int):
        self._gamma = ueye.UINT(value)
        cam_gamma = ueye.UINT()
        nRet = ueye.is_Gamma(self.hCam, ueye.IS_GAMMA_CMD_SET, self._gamma, ueye.sizeof(self._gamma))
        if nRet != ueye.IS_SUCCESS:
            print("is_Gamma ERROR")
        print(f"Set Gamma: \t{self._gamma / 100:.2f}")
        nRet = ueye.is_Gamma(self.hCam, ueye.IS_GAMMA_CMD_GET, cam_gamma, ueye.sizeof(cam_gamma))
        if nRet != ueye.IS_SUCCESS:
            print("is_Gamma ERROR")
        print(f"Current Gamma: \t{cam_gamma / 100:.2f}")

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, value: int):
        self._gain = ueye.UINT(value)  # 100 = gain factor x1.00, i.e. no effect
        leo_temp = ueye.UINT()
        nRet = ueye.is_SetHWGainFactor(self.hCam, ueye.IS_SET_MASTER_GAIN_FACTOR, self._gain)
        print(f"Set Gain: \tx{nRet / 100:.2f}")
        nRet = ueye.is_SetHWGainFactor(self.hCam, ueye.IS_GET_MASTER_GAIN_FACTOR, leo_temp)
        print(f"Current Gain: \tx{nRet / 100:.2f}")


    def capture(self, mean: int = 1):
        ## Under software trigger mode, take a snap
        nRet = ueye.is_FreezeVideo(self.hCam, ueye.IS_WAIT)  # IS_WAIT (When 1st image in memory), IS_DONT_WAIT (the functions returns immediately, so you have to use the API event IS_SET_EVENT_FRAME for querying if the image is available in the memory)
        if nRet != ueye.IS_SUCCESS:
            print("is_FreezeVideo ERROR", nRet)
        
        ## Extract the data of our image memory
        array = ueye.get_data(self.image_memory, self.aoi_width, self.aoi_height, self.n_bits_per_pixel, self.pitch, copy=True)
    
        ## Each element is a byte (8 bit), one pixel (12bit) is two elements
        if self.n_bits_per_pixel > 8:
            array = array[0::2] + array[1::2]*256  # Byte1 is 15~8bit, Byte0 is 7~0bit, see "Color and memory formats" in iDS uEye manual
    
        ## reshape it into an numpy array
        if self.n_channels == 1:
            frame = np.reshape(array, (self.aoi_height.value, self.aoi_width.value))  # bytes_per_pixel = int(nBitsPerPixel / 8)
        else:
            frame = np.reshape(array, (self.aoi_height.value, self.aoi_width.value, self.n_channels))
        
        ## if needed, Do average
        if mean > 1:
            sum_frame = frame.copy().astype(dtype=np.float64)  # use float ∵ int /= int will have problem
            for i in range(mean-1):
                sum_frame += self.capture()
            sum_frame /= mean
            return sum_frame
        return frame

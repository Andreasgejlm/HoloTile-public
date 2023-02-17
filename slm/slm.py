import numpy as np
try:
    import detect_heds_module_path
    from holoeye import slmdisplaysdk
except ImportError:
    print("Holoeye heds not found")

class SLM:
    def __init__(self):
        try:
            self.slm_object = slmdisplaysdk.SLMInstance()
            error = self.slm_object.open()
            if not error == slmdisplaysdk.ErrorCode.NoError:
                self.slm_object.errorString(error)
                self.slm_object = None
                self.W = self.H = self.M = self.N = None
            else:
                self.W = self.slm_object.width_mm * 1E-3
                self.H = self.slm_object.height_mm * 1E-3
                self.M = self.slm_object.width_px
                self.N = self.slm_object.height_px
                self.lp = self.W / self.M
            print("SLM open with", (self.W, self.H, self.M, self.N))
        except (AttributeError, RuntimeError) as e:
            self.slm_object = None
            self.W = self.H = self.M = self.N = None
        self.displayOptions = slmdisplaysdk.ShowFlags.PresentAutomatic  # PresentAutomatic == 0 (default)
        self.displayOptions |= slmdisplaysdk.ShowFlags.PresentFitWithBars

    def show_gray_value_image(self, phase_image: np.ndarray):
        assert np.max(phase_image) < 256, "Phase image has too large values (>255)"
        if self.slm_object is not None:
            self.slm_object.showData(phase_image)
        else:
            print("SLM object is None")

    def show_image_from_file(self, filepath: str):
        if self.slm_object is not None:
            self.slm_object.showDataFromFile(filepath, self.displayOptions)
        else:
            print("SLM object is None")

    def show_phase(self, phase_image: np.ndarray) -> None:
        if self.slm_object is not None:
            self.slm_object.showPhasevalues(phase_image)
        else:
            print("SLM object is None")

    def get_coordinates(self):
        return np.meshgrid(np.linspace(-self.W/2, self.W/2, self.M), np.linspace(-self.H/2, self.H/2, self.N))

    def get_dimensions(self):
        return self.H, self.W

    def get_pixels(self):
        return self.N, self.M


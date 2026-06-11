import torchattacks

def apply_cw_patch():
    original_cw_f = torchattacks.CW.f
    def patched_cw_f(self, outputs, labels):
        return original_cw_f(self, outputs, labels.cpu())
    torchattacks.CW.f = patched_cw_f
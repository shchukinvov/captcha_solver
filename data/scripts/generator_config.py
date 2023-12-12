"""
Config for captcha_generator describing noising params.
Characters and numbers list.
List of fonts.
"""
NOISE_PARAMS = {
    "bg_params": {
        "low": {
            "scale": 0.25,
            "frequency_low": 10,
            "frequency_high": 1
        },
        "medium": {
            "scale": 0.4,
            "frequency_low": 25,
            "frequency_high": 2
        },
        "high": {
            "scale": 0.75,
            "frequency_low": 50,
            "frequency_high": 5
        }
    },
    "curve_params": {
        "low": {
            "num_lines": 2,
            "thickness": 1,
            "frequency": 15,
        },
        "medium": {
            "num_lines": 2,
            "thickness": 2,
            "frequency": 15,
        },
        "high": {
            "num_lines": 3,
            "thickness": 2,
            "frequency": 10,
        }
    }
}

CHARACTERS = "abcdefghiklmnopqrstvxyzABCDEFGHIKLMNOPQRSTVXYZ"
NUMS = "123456789"
FONTS = [0, 2, 3, 4, 16]
"""
More details in https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html
FONT_HERSHEY_SIMPLEX        = 0,
FONT_HERSHEY_PLAIN          = 1,
FONT_HERSHEY_DUPLEX         = 2,
FONT_HERSHEY_COMPLEX        = 3,
FONT_HERSHEY_TRIPLEX        = 4,
FONT_HERSHEY_COMPLEX_SMALL  = 5,
FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
FONT_HERSHEY_SCRIPT_COMPLEX = 7,
FONT_ITALIC                 = 16
"""

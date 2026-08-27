import numpy as np

cpc = "adi"

headers = {
    "dma": {
        "Time": "datetime",
        "DMA Voltage": "dma_voltage",
        "Electrometer Concentration": "concentration",
        "Time Since Start": "time_since_start",
        "Electrometer Voltage": "voltage",
        "DMA Set Voltage": "dma_set_voltage",
    },
    # "adi": [
    #     "datetime",
    #     "instrument_datetime",
    #     "concentration",
    #     "temp_conditioner",
    #     "temp_initiator",
    #     "temp_moderator",
    #     "temp_optics",
    #     "temp_heatsink",
    #     "temp_pcb",
    #     "supply_voltage",
    #     "diff_press",
    #     "abs_press",
    #     "flow_rate",
    #     "time_interval",
    #     "time_corrected_live",
    #     "time_dead",
    #     "raw_counts_low",
    #     "raw_counts_high",
    #     "flags",
    #     "errors",
    #     "serial_number",
    # ],
    "adi":["cpc name",
           "datetime",
           "instrument_datetime",
           "concentration",
           "condenser_temperature",
           "initiator_temperature",
           "moderator_temperature",
           "optics_temperature",
           "moderator_heatsink_temperature",
           "board_temperature",
           "condenser_heatsink_temperature",
           "differential_pressure",
           "absolute_pressure",
           "flow_rate",
           "interval_time",
           "corrected_live_time",
           "measured_dead_time",
           "xx",
           "xxx",
           "flags",
           "flags_character",
           "serial_number"],
    "tsi": [
        "cpc_name",
        "datetime",
        "concentration",
        "condensor_temp",
        "saturator_temp",
        "optics_temp",
        "flow",
        "ready_environment",
        "reference_detector_voltage",
        "detector_voltage",
        "pump_control_value",
        "one_sec_counts",
        "liquid_level"
    ],
}

datetime_col = {
    "adi": "datetime",
    "tsi": "datetime"
}

read_settings = {
    "dma": {
        "filetype": ("CSV Files", "DMA*avg.csv*"),
        "datecol": "datetime",
        "tzone": "US/Eastern",
    },
    # "adi": {
    #     #     "filetype": ("Text Files", "MAGIC*.txt*"),
    #     #     "filepattern": "MAGIC*.txt",
    #     #     "datecol": "datetime",
    #     #     "tzone": "US/Eastern",
    #     # },
    "adi": {
            "filetype": ("CSV Files", "*.csv*"),
            "filepattern": "MANY*.csv",
            "datecol": "datetime",
            "tzone": "US/Eastern",
        },
    "tsi": {
        "filetype": ("CSV Files", "*.csv"),
        "filepattern": "MANY*.csv",  # optional
        "datecol": "datetime",
        "tzone": "US/Eastern",
    },
}

fit_settings = {"bounds": ([0, 0.1, 0], [1, np.inf, np.inf])}

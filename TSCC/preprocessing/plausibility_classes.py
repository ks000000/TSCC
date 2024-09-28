'''
Plausible value check from Zahumensky:
Guidelines on Quality Control Procedures for Data from Automatic Weather Stations
'''

#plausibleValueRange
class rangePlausVals:
    '''
    TODO: add option to change/overwrite values to local conditions
    '''

    dict = {
        "air_temperature": {
            "lower_boundary": -80,
            "upper_boundary": 60,
            "unit": "°C",
            "additional_information": ""
        },
        "dew_point_temperature": {
            "lower_boundary": -80,
            "upper_boundary": 35,
            "unit": "°C",
            "additional_information": ""
        },
        "ground_temperature": {
            "lower_boundary": -80,
            "upper_boundary": 80,
            "unit": "°C",
            "additional_information": ""
        },
        "soil_temperature": {
            "lower_boundary": -50,
            "upper_boundary": 50,
            "unit": "°C",
            "additional_information": ""
        },
        "relative_humidity": {
            "lower_boundary": 0,
            "upper_boundary": 100,
            "unit": "%",
            "additional_information": ""
        },
        "pressure": {
            "lower_boundary": 500,
            "upper_boundary": 1100,
            "unit": "hPa",
            "additional_information": "Atmospheric pressure at the station level"
        },
        "water_temperature": {
            "lower_boundary": 0,
            "upper_boundary": 100,
            "unit": "°C",
            "additional_information": "water in liquid state"
        },
        "wind_direction": {
            "lower_boundary": 0,
            "upper_boundary": 360,
            "unit": "degree",
            "additional_information": ""
        },
        "wind_speed": {
            "lower_boundary": 0,
            "upper_boundary": 75,
            "unit": "m/s^(-1)",
            "additional_information": "2-minute, 10-minute average"
        },
        "solar_radiation": {
            "lower_boundary": 0,
            "upper_boundary": 1600,
            "unit": "Wm^(-2)",
            "additional_information": "irradiance"
        },
        "precipitation": {
            "lower_boundary": 0.0,
            "upper_boundary": 40.0,
            "unit": "mm",
            "additional_information": "amount in 1 minute time interval"
        },
        "discharge": {
            "lower_boundary": 0,
            "upper_boundary": 1000000,
            "unit": "l/sec",
            "additional_information": "flow in sewer system, very broad upper limit - real limit greatly depends on local "
                                      "conditions"
        }
    }

class plausibleValueStep:
    # consecutive observations
    plausibleValuesDict = {
        "air_temperature": {
            "max": 3,
            "max_sec": 2,
            "unit": "°C",
            "additional_information": "in one-minute interval"
        },
        "dew_point_temperature": {
            "max": 4,
            "max_sec": 2,
            "unit": "°C",
            "additional_information": "in one-minute interval, limit for suspect can start from 2 on"
        },
        "ground_temperature": {
            "max": 10,
            "max_sec": 2,
            "unit": "°C",
            "additional_information": "in one-minute interval, limit for suspect starts from 5 on"
        },
        "soil_temperature": {
            "max": 0.1,
            "unit": "°C",
            "additional_information": "in one-minute interval, can range down to 0.2 depending on depth of measurement station"
        },
        "relative_humidity": {
            "max": 15,
            "max_sec": 5,
            "unit": "%",
            "additional_information": "in one-minute interval"
        },
        "pressure": {
            "max": 2,
            "max_sec": 0.3,
            "unit": "hPa",
            "additional_information": "in one-minute interval, limit for suspect from 0.5 on"
        },
        "wind_speed": {
            "max": 20,
            "max_sec": 20,
            "unit": "m/s",
            "additional_information": "in one-minute interval, limit for suspect from 10 on"
        },
        "solar_radiation": {
            "max": 1000,
            "max_sec": 800,
            "unit": "m/s",
            "additional_information": "in one-minute interval, limit for suspect from 800 on"
        }
    }

class plausibleValueVariability:

    dict = {
        "air_temperature": {
            "min": 0.1,
            "unit": "°C",
            "additional_information": "time interval at least 60 minutes for one value per minute"
        },
        "dew_point_temperature": {
            "min": 0.1,
            "unit": "°C",
            "additional_information": "time interval at least 60 minutes for one value per minute"
        },
        "ground_temperature": {
            "min": 0.1,
            "unit": "°C",
            "additional_information": "time interval at least 60 minutes for one value per minute,"
                                      "for ground temperature outside the interval [-0.2, 0.2]"
        },
        "relative_humidity": {
            "min": 1,
            "unit": "%",
            "additional_information": "time interval at least 60 minutes for one value per minute,"
                                      "for relative humidity less than 95"
        },
        "pressure": {
            "min": 0.1,
            "unit": "hPa",
            "additional_information": "time interval at least 60 minutes for one value per minute"
        },
        "wind_direction": {
            "min": 10,
            "unit": "degree",
            "additional_information": "time interval at least 60 minutes for one value per minute,"
                                      "For 10-minute average wind speed during the period > 0.1 ms"
        },
        "wind_speed": {
            "min": 0.5,
            "unit": "m/s",
            "additional_information": "time interval at least 60 minutes for one value per minute,"
                                      "For 10-minute average wind speed during the period > 0.1 ms"
        }
    }


class plausibleValue_internalConsistency:
    #The basic algorithms used for checking internal consistency of data are based on the relation between two parameters (the following conditions shall be true):
    # failing values should be flagged as error at plausibleValuesDict!
    # failing values should be flagged as suspect at unplausibleValuesDict!
    # unplausibleValuesDict only for data in time range of 15 minutes max

    plausibleValuesDict = {
        "air_temperature": {
            "relation": {
                "kind": "lessorequal",
                "parameter": "dew_point_temperature",
            },
            "additional_information": ""
        },
        "dew_point_temperature": {
            "relation": {
                "kind": "greaterorequal",
                "parameter": "air_temperature",
            },
            "additional_information": ""
        },
        "wind_speed": {
            "relation": {
                "kind": "equal",
                "val_range": 0,
                "vale_range_other_param": 0,
                "parameter": "wind_direction"
            },
            "relation": {
                "kind": "equal",
                "val_range":(0,float("inf")),
                "vale_range_other_param": (0,float("inf")),
                "parameter": "wind_direction"
            },
            "additional_information": ""
        },
        "wind_direction": {
            "relation": {
                "kind": "equal",
                "val_range": 0,
                "vale_range_other_param": 0,
                "parameter": "wind_speed"
            },
            "relation": {
                "kind": "equal",
                "val_range":(0,float("inf")),
                "vale_range_other_param": (0,float("inf")),
                "parameter": "wind_speed"
            },
            "additional_information": ""
        },
    }
    #wind_gust(speed)>=wind_speed


    UnplausibleValuesDict = {
        "total_cloud_cover": {
            "relation": {
                "kind": "equal",
                "val_range": 0,
                "vale_range_other_param": (0, float("inf")),
                "parameter": "precipitation"
            },
            "relation": {
                "kind": "equal",
                "val_range": 8,
                "vale_range_other_param": (0, float("inf")),
                "parameter": "sunshine_duration"
            },
            "additional_information": ""
        },
        "sunshine_duration": {
            "relation": {
                "kind": "equal",
                "val_range": (0, float("inf")),
                "vale_range_other_param": 0,
                "parameter": "solar_radiation"
            },
            "relation": {
                "kind": "equal",
                "val_range": 0,
                "vale_range_other_param": (500, float("inf")),
                "parameter": "solar_radiation"
            },
            "relation": {
                "kind": "equal",
                "val_range": (0, float("inf")),
                "vale_range_other_param": 8,
                "parameter": "total_cloud_cover"
            },
            "additional_information": "total cloudy and sunshine"
        },
        "precipitation": {
            "relation": {
                "kind": "equal",
                "val_range": (0, float("inf")),
                "vale_range_other_param": 0,
                "parameter": "total_cloud_cover"
            },
            "additional_information": "precipitation amount >0 and no cloud"
        },
    }

    #total_cloud_cover == 0 and precipitation_duration > 0 #precipitation time >0 and no cloud
    #precipitation > 0 and precipitation_duration == 0

class sensorUncertainty:
    sensorUncertainty = {
        "air_temperature": {
            "low_cost": {
                "accuracy": 0.5,
                "unit": "°C",
                "additional_information": ""
            },
            "high_cost": {
                "accuracy": 0.05,
                "unit": "°C",
                "additional_information": ""
            }
        },
        "dew_point_temperature": {
            "low_cost": {
                "accuracy": 1.0,
                "unit": "°C",
                "additional_information": ""
            },
            "high_cost": {
                "accuracy": 0.2,
                "unit": "°C",
                "additional_information": ""
            }
        },
        "ground_temperature": {
            "low_cost": {
                "accuracy": 0.5,
                "unit": "°C",
                "additional_information": ""
            },
            "high_cost": {
                "accuracy": 0.1,
                "unit": "°C",
                "additional_information": ""
            }
        },
        "soil_temperature": {
            "low_cost": {
                "accuracy": 1.0,
                "unit": "°C",
                "additional_information": ""
            },
            "high_cost": {
                "accuracy": 0.2,
                "unit": "°C",
                "additional_information": ""
            }
        },
        "relative_humidity": {
            "low_cost": {
                "accuracy": 3.0,
                "unit": "%",
                "additional_information": ""
            },
            "high_cost": {
                "accuracy": 1.0,
                "unit": "%",
                "additional_information": ""
            }
        },
        "pressure": {
            "low_cost": {
                "accuracy": 1.0,
                "unit": "bar",
                "additional_information": ""
            },
            "high_cost": {
                "accuracy": 0.1,
                "unit": "bar",
                "additional_information": ""
            }
        },
        "water_temperature": {
            "low_cost": {
                "accuracy": 0.5,
                "unit": "°C",
                "additional_information": ""
            },
            "high_cost": {
                "accuracy": 0.1,
                "unit": "°C",
                "additional_information": ""
            }
        },
        "wind_direction": {
            "low_cost": {
                "accuracy": 5.0,
                "unit": "degrees",
                "additional_information": ""
            },
            "high_cost": {
                "accuracy": 1.0,
                "unit": "degrees",
                "additional_information": ""
            }
        },
        "wind_speed": {
            "low_cost": {
                "accuracy": 0.5,
                "unit": "m/s",
                "additional_information": ""
            },
            "high_cost": {
                "accuracy": 0.1,
                "unit": "m/s",
                "additional_information": ""
            }
        },
        "solar_radiation": {
            "low_cost": {
                "accuracy": 10.0,
                "unit": "W/m²",
                "additional_information": ""
            },
            "high_cost": {
                "accuracy": 1.0,
                "unit": "W/m²",
                "additional_information": ""
            }
        },
        "precipitation": {
            "low_cost": {
                "accuracy": 1.0,
                "unit": "mm/h",
                "additional_information": "netatmo rain gauge as reference"
            },
            "high_cost": {
                "accuracy": 0.1,
                "unit": "mm",
                "additional_information": "ott pluvio as reference, 0.1mm or 1%"
            }
        },
        "discharge": {
            "low_cost": {
                "accuracy": 5.0,
                "unit": "m³/s",
                "additional_information": ""
            },
            "high_cost": {
                "accuracy": 0.1,
                "unit": "m³/s",
                "additional_information": ""
            }
        }
    }



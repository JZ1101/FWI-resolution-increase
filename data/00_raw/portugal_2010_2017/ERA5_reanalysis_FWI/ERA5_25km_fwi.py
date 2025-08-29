import cdsapi

dataset = "cems-fire-historical-v1"
request = {
    "product_type": "reanalysis",
    "variable": ["fire_weather_index"],
    "dataset_type": "consolidated_dataset",
    "system_version": ["4_1"],
    "year": ["2017"],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "grid": "0.25/0.25",
    "data_format": "grib"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()

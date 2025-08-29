import cdsapi

dataset = "reanalysis-uerra-europe-single-levels"
request = {
    "origin": "mescan_surfex",
    "variable": "total_precipitation",
    "year": [
        "2010", "2011", "2012",
        "2013", "2014", "2015",
        "2016", "2017"
    ],
    "month": [
        "05", "06", "07",
        "08", "09", "10",
        "11"
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
    "time": ["06:00"],
    "data_format": "netcdf"
}

# Specify the output filename
output_filename = "uerra_total_precipitation_2010_2017_may_nov_06h.nc"

client = cdsapi.Client()
client.retrieve(dataset, request, output_filename)

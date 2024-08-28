from q4l.data.factorhub import FactorHubAPI

flist = [
    "adj_volume_1min",
    "adj_vwap_1min",
    "adj_open_px_1min",
    "adj_close_px_1min",
    "adj_high_px_1min",
    "adj_low_px_1min",
]

apii = FactorHubAPI("wangsaizhuo", "3mbeR_5p1RiT")


for factor in flist:
    df = apii.get_factor(factor, start_time="2015-01-01", end_time="2023-08-01")
    df.to_csv(f"{factor}_cn1min.csv")

import os
from icrawler.builtin import GoogleImageCrawler


date_ranges = [
    {'date': ((2013, 1, 1), (2013, 6, 30))},
    {'date': ((2013, 6, 30), (2013, 12, 31))},

    {'date': ((2014, 1, 1), (2014, 6, 30))},
    {'date': ((2014, 6, 30), (2014, 12, 31))},

    {'date': ((2015, 1, 1), (2015, 6, 30))},
    {'date': ((2015, 6, 30), (2015, 12, 31))},

    {'date': ((2016, 1, 1), (2016, 6, 30))},
    {'date': ((2016, 6, 30), (2016, 12, 31))},

    {'date': ((2017, 1, 1), (2017, 6, 30))},
    {'date': ((2017, 6, 30), (2017, 12, 31))},

    {'date': ((2018, 1, 1), (2018, 6, 30))},
    {'date': ((2018, 6, 30), (2018, 12, 31))},

    {'date': ((2019, 1, 1), (2019, 6, 30))},
    {'date': ((2019, 6, 30), (2019, 12, 31))},

    {'date': ((2020, 1, 1), (2020, 6, 30))},
    {'date': ((2020, 6, 30), (2020, 12, 31))},

    {'date': ((2021, 1, 1), (2021, 6, 30))},
    {'date': ((2021, 6, 30), (2021, 12, 31))},

    {'date': ((2022, 1, 1), (2022, 6, 30))},
    {'date': ((2022, 6, 30), (2022, 12, 31))},

    {'date': ((2023, 1, 1), (2023, 6, 30))},
    {'date': ((2023, 6, 30), (2023, 12, 31))},
]

targets = {
    'Calimerio': {
        'keyword': 'cúc calimero',
        'amount': 600
    },
    'Chrysanthemum': {
        'keyword': 'chrysanthemum flower',
        'amount': 214
    },
    'Hydrangeas': {
        'keyword': 'hoa tú cầu',
        'amount': 214
    },
    'Pingpong': {
        'keyword': 'cúc ping pong',
        'amount': 540
    },
    'Rosy': {
        'keyword': 'cúc rossi',
        'amount': 730
    },
    'Tana': {
        'keyword': 'hoa cúc tana',
        'amount': 280
    },
}


if __name__ == '__main__':
    for flower_class, scrape_config in targets.items():
        bot = GoogleImageCrawler(
            storage={'root_dir': flower_class},
            feeder_threads=1,
            parser_threads=1,
            downloader_threads=4,
        )

        for iteration, date_range in enumerate(date_ranges):
            bot.crawl(
                keyword=scrape_config['keyword'],
                max_num=scrape_config['amount'],
                filters=date_range,
                file_idx_offset=0 if iteration == 0 else 'auto',
            )

            if len(os.listdir(flower_class)) >= scrape_config['amount']:
                break

        if len(os.listdir(flower_class)) >= scrape_config['amount']:
            continue

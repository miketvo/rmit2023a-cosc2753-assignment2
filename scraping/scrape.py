import datetime
import os
from icrawler.builtin import GoogleImageCrawler


targets = {
    'Baby': {
        'keyword': 'baby\'s breath flower',
        'amount': 1069
    },
    'Calimerio': {
        'keyword': 'cúc calimero',
        'amount': 1647
    },
    'Chrysanthemum': {
        'keyword': 'chrysanthemum flower',
        'amount': 1304
    },
    'Hydrangeas': {
        'keyword': 'hoa tú cầu',
        'amount': 1482
    },
    'Lisianthus': {
        'keyword': 'bó hoa cát tường',
        'amount': 1031
    },
    'Pingpong': {
        'keyword': 'cúc ping pong',
        'amount': 1640
    },
    'Rosy': {
        'keyword': 'cúc rossi',
        'amount': 1829
    },
    'Tana': {
        'keyword': 'hoa cúc tana',
        'amount': 1377
    },
}


def date_ranges(start_datetime, interval, count):
    results = []
    for i in range(count):
        end_date = start_datetime - datetime.timedelta(days=i * interval)  # Calculate the end date
        start_date = end_date - datetime.timedelta(days=interval)  # Calculate the start date
        result = {
            'date': (
                (start_date.year, start_date.month, start_date.day),
                (end_date.year, end_date.month, end_date.day)
            )
        }
        results.append(result)
    return results


if __name__ == '__main__':
    for iteration, date_range in enumerate(date_ranges(datetime.datetime.now().date(), interval=7, count=156)):
        for flower_class, scrape_config in targets.items():
            if iteration > 0 and len(os.listdir(flower_class)) >= scrape_config['amount']:
                continue

            bot = GoogleImageCrawler(
                storage={'root_dir': flower_class},
                feeder_threads=1,
                parser_threads=1,
                downloader_threads=4,
            )

            bot.crawl(
                keyword=scrape_config['keyword'],
                max_num=scrape_config['amount'],
                filters=date_range,
                file_idx_offset=0 if iteration == 0 else 'auto',
            )

    print('\n[==== DONE ====]')
